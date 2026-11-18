"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), {spawnSync} = require("node:child_process");
const patch = fs.readFileSync(path.join(__dirname, "../runtime/dpcpp-static-adapters.patch"), "utf8");
/** Recover changed original source including context, never removed lines or patch headers.
 * @param {string} name */
function source(name) {
  const block = patch.split(/(?=^diff --git )/m).find(part => part.startsWith(`diff --git a/${name} b/${name}\n`));
  assert.ok(block, "Missing original source patch: " + name);
  return block.split("\n").filter(line => /^[ +]/.test(line) && !line.startsWith("+++"))
    .map(line => line.slice(1)).join("\n");
}

test.describe("Native static adapter selection without hardware", {skip: process.platform === "win32"}, () => {
  let root = "", executable = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-static-adapters-")); executable = path.join(root, "selection");
    const registry = source("unified-runtime/source/loader/ur_adapter_registry.hpp");
    const method = registry.slice(registry.indexOf("  bool isAdapterEnabled("), registry.indexOf("  std::vector<"));
    const loader = source("unified-runtime/source/loader/ur_loader.cpp");
    const start = loader.indexOf("#if defined(UR_STATIC_ADAPTER_LEVEL_ZERO)");
    assert.ok(method.includes("p.filename() == name") && start >= 0);
    // Exercise the actual registration block against already-filtered manifest identities.
    // The opt-in SDK fixture separately tests the real upstream environment parser.
    const code = `#include <algorithm>\n#include <filesystem>\n#include <vector>\n#include <string>\n#include <cstdlib>\n
namespace fs=std::filesystem;
#define UR_STATIC_ADAPTER_LEVEL_ZERO
#define UR_STATIC_ADAPTER_LEVEL_ZERO_V2
#define UR_STATIC_ADAPTER_CUDA
#define UR_STATIC_ADAPTER_OPENCL
#define MAKE_LIBRARY_NAME(name,version) "lib" name ".so." version
enum {UR_RESULT_SUCCESS=0}; using ur_result_t=int;
static unsigned actual=0;static bool failCuda=false;
namespace ur::level_zero {int urAdapterGetDdiTables(int*){actual|=1;return 0;}}
namespace ur::opencl {int urAdapterGetDdiTables(int*){actual|=4;return 0;}}
int mom_ur_cuda_factory(int*){actual|=2;return failCuda?89:0;}
int mom_ur_level_zero_v2_factory(int*){actual|=8;return 0;}
struct Platform {int dditable;explicit Platform(void*):dditable(0){}};
struct Registry {
  bool forced=false;std::vector<std::vector<fs::path>> adaptersLoadPaths;
  bool adaptersForceLoaded()const{return forced;}
  void markAdapterAsStaticallyLoaded(const char*){}
${method}
};
static Registry adapter_registry;static std::vector<Platform> platforms;
int bootstrap(){
${loader.slice(start)}
return 0;}
int main(int argc,char** argv){
  if(argc!=5)return 85;
  unsigned wanted=unsigned(std::atoi(argv[1]));
  adapter_registry.forced=std::atoi(argv[2])!=0;failCuda=std::atoi(argv[3])!=0;
  const char* names[]={"libur_adapter_level_zero.so.0","libur_adapter_cuda.so.0","libur_adapter_opencl.so.0","libur_adapter_level_zero_v2.so.0"};
  for(unsigned i=0;i<4;++i)if(wanted&(1u<<i))adapter_registry.adaptersLoadPaths.push_back({fs::path("/sdk")/names[i]});
  adapter_registry.adaptersLoadPaths.push_back({"/sdk/libur_adapter_opencl.so.0.untrusted"});
  int result=bootstrap();unsigned expected=adapter_registry.forced?0:wanted;
  if(result!=std::atoi(argv[4]))return 86;
  return actual==expected?0:87;
}\n`;
    fs.writeFileSync(path.join(root, "selection.cpp"), code);
    const built = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      path.join(root, "selection.cpp"), "-o", executable], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {number} mask @param {boolean} force @param {boolean} fail @param {number} expected */
  function check(mask, force, fail, expected) {
    const run = spawnSync(executable, [String(mask), force ? "1" : "0", fail ? "1" : "0", String(expected)], {encoding: "utf8", timeout: 5000});
    assert.equal(run.status, 0, run.stderr);
  }
  for (let mask = 0; mask < 8; ++mask) {
    test(`preparsed mask ${mask} registers exactly its selected factories`, () => {check(mask, false, false, 0);});
  }
  for (const mask of [0, 1, 7]) {
    test(`forced adapter loading suppresses static mask ${mask}`, () => {check(mask, true, false, 0);});
  }
  test("both original Level Zero versions may register while other backends stay excluded", () => {check(9, false, false, 0);});
  test("the unfiltered registry retains all linked adapter versions", () => {check(15, false, false, 0);});
  test("an excluded failing CUDA factory is never invoked", () => {check(1, false, true, 0);});
  test("a selected factory failure is propagated", () => {check(2, false, true, 89);});
  test("the same original manifest filtering serves every static profile", () => {
    const helper = fs.readFileSync(path.join(__dirname, "../scripts/native-dpcpp-pins.js"), "utf8");
    assert.match(helper, /adapter \|\| cudaSdk \|\| opencl \? \["dpcpp-static-adapters\.patch"\]/);
    assert.doesNotMatch(patch, /getenv\(|getenv_tobool\(|getenv_to_map\(/);
  });
});
