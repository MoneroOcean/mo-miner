"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {isolateLevelZeroAdapter} = require("../scripts/native-level-zero-adapter");
const repo = path.resolve(__dirname, ".."), read = (/** @type {string} */ name) => fs.readFileSync(path.join(repo, name), "utf8");

test.describe("Native Level Zero V2 original-code isolation", {skip: process.platform !== "linux"}, () => {
  test("importing the producer cannot load a driver or begin compilation", () => {
    const run = spawnSync(process.execPath, ["-e", 'require("./scripts/native-level-zero-adapter")'],
      {cwd: repo, env: {...process.env, PATH: "/unavailable"}, encoding: "utf8", timeout: 5000});
    assert.equal(run.status, 0, run.stderr); assert.equal(run.stdout, ""); assert.equal(run.stderr, "");
  });
  test("missing build-only admission rejects before probing inputs", (t) => {
    const before = process.env["MOM_AOT_BUILD_ONLY"]; delete process.env["MOM_AOT_BUILD_ONLY"];
    t.after(() => {if (before === undefined) {delete process.env["MOM_AOT_BUILD_ONLY"];} else {process.env["MOM_AOT_BUILD_ONLY"] = before;}});
    assert.throws(() => isolateLevelZeroAdapter("/missing/archive", "/missing/cxx", "/missing/headers", "/missing/output"), /MOM_AOT_BUILD_ONLY/);
  });
  test("the original V2 implementation and upstream device-version policy are not rewritten", () => {
    const patch = read("runtime/dpcpp-static-adapters.patch");
    assert.match(patch, /set\(ADAPTER_V2_LIB_TYPE STATIC\)/);
    assert.match(patch, /target_compile_definitions\(ur_adapter_level_zero_v2 PRIVATE UR_STATIC_ADAPTER_LEVEL_ZERO\)/);
    assert.doesNotMatch(patch, /^diff --git .*level_zero\/.*\.(?:cpp|hpp)/m);
    assert.doesNotMatch(patch, /isBMGorNewer|shouldUseV[12]|SYCL_UR_USE_LEVEL_ZERO_V2/);
  });
  test("the opaque bridge reuses the original complete table and validates null", () => {
    const factory = read("tests/native/dpcpp-static/level-zero-v2-factory.cpp");
    assert.match(factory, /table \? ur::level_zero::urAdapterGetDdiTables\(table\) : UR_RESULT_ERROR_INVALID_NULL_POINTER/);
    assert.doesNotMatch(factory, /pfn\w+\s*=|getenv|dlopen/);
    const helper = read("scripts/native-level-zero-adapter.js");
    assert.match(helper, /isolateCompilerObject\(/);
    assert.match(helper, /entryPoints: \["mom_ur_level_zero_v2_factory"\]/);
  });
  test("CMake retains both original adapter graphs and their transitive dependencies", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    for (const name of ["UR_BUILD_ADAPTER_L0", "UR_BUILD_ADAPTER_L0_V2"]) {assert.ok(cmake.includes(`set(${name} ON CACHE BOOL "" FORCE)`));}
    assert.match(cmake, /get_target_property\(l0_v2_dependencies ur_adapter_level_zero_v2 LINK_LIBRARIES\)/);
    assert.match(cmake, /target_link_libraries\(mom-ur-level-zero-v2-static INTERFACE "\$\{l0_v2_object\}" \$\{l0_v2_dependencies\}\)/);
  });
  test("the original no-driver proof checks V2's complete table and absent-driver result", () => {
    const proof = read("tests/reference/native_dpcpp_static.cpp");
    assert.match(proof, /mom_ur_level_zero_v2_factory\(nullptr\) != UR_RESULT_ERROR_INVALID_NULL_POINTER/);
    assert.match(proof, /!v2\.EnqueueExp\.pfnKernelLaunchWithArgsExp/);
    assert.match(proof, /v2\.Adapter\.pfnGet\(0, nullptr, &count\) != UR_RESULT_ERROR_UNSUPPORTED_VERSION \|\| count/);
  });
  test("real bridge isolation preserves opaque calls without colliding with V1 names", (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-l0-v2-bridge-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const headers = path.join(root, "headers"); fs.mkdirSync(path.join(headers, "unified-runtime"), {recursive: true});
    fs.writeFileSync(path.join(headers, "unified-runtime/ur_api.h"), "#pragma once\nenum ur_result_t {UR_RESULT_SUCCESS,UR_RESULT_ERROR_INVALID_NULL_POINTER};\n");
    fs.writeFileSync(path.join(headers, "unified-runtime/ur_ddi.h"), "#pragma once\nstruct ur_dditable_t {int(*call)(int);};\n");
    fs.writeFileSync(path.join(root, "original.cpp"), "#include <unified-runtime/ur_api.h>\n#include <unified-runtime/ur_ddi.h>\n" +
      "namespace ur::level_zero {static int operation(int x){return x+19;} ur_result_t urAdapterGetDdiTables(ur_dditable_t* t){t->call=operation;return UR_RESULT_SUCCESS;}}\n");
    /** @param {string} command @param {string[]} args */
    function run(command, args) {
      const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.stderr); return result.stdout.trim();
    }
    const cxx = run("sh", ["-c", "command -v \"$1\"", "--", process.env["CXX"] || "c++"]);
    run(cxx, ["-std=c++17", "-O2", "-fPIC", "-I", headers, "-c", path.join(root, "original.cpp"), "-o", path.join(root, "original.o")]);
    const archive = path.join(root, "original.a"); run("ar", ["crs", archive, path.join(root, "original.o")]);
    const before = process.env["MOM_AOT_BUILD_ONLY"]; process.env["MOM_AOT_BUILD_ONLY"] = "1";
    let isolated;
    try {isolated = isolateLevelZeroAdapter(archive, cxx, headers, path.join(root, "private"));} finally {
      if (before === undefined) {delete process.env["MOM_AOT_BUILD_ONLY"];} else {process.env["MOM_AOT_BUILD_ONLY"] = before;}
    }
    fs.writeFileSync(path.join(root, "check.cpp"), "#include <unified-runtime/ur_api.h>\n#include <unified-runtime/ur_ddi.h>\n" +
      'extern "C" ur_result_t mom_ur_level_zero_v2_factory(ur_dditable_t*);\n' +
      "namespace ur::level_zero {ur_result_t urAdapterGetDdiTables(ur_dditable_t*){return UR_RESULT_ERROR_INVALID_NULL_POINTER;}}\n" +
      "int main(){ur_dditable_t t{};return mom_ur_level_zero_v2_factory(nullptr)!=UR_RESULT_ERROR_INVALID_NULL_POINTER||" +
      "mom_ur_level_zero_v2_factory(&t)!=UR_RESULT_SUCCESS||!t.call||t.call(23)!=42;}\n");
    const executable = path.join(root, "check");
    run(cxx, ["-std=c++17", "-I", headers, path.join(root, "check.cpp"), isolated.object, "-o", executable]); run(executable, []);
  });
});
