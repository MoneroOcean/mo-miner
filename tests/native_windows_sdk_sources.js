"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {sdkComponents, sdkPatchScope, validateSourceRoots, sdkSourceEnvironment, sdkSourceFiles, verifySdkExport, prepareSdkSources, verifyPreparedSdkSources} = require("../scripts/prepare-native-sdk-sources");
const components = sdkComponents("windows");
const {pins} = require("../scripts/probe-static-dpcpp"), {fileHash} = require("../scripts/build-source-inputs");

/** Isolated source fixtures never use installed SDKs, network, compilers, VMs or GPUs.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sdk-sources-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const source = path.join(root, "source"); fs.mkdirSync(source);
  return {root, source, output: path.join(root, "output"), roots: Object.fromEntries(Object.keys(components).map(key => [key, source]))};
}

test.describe("Pinned source-JIT compiler source profiles", () => {
  test("compiler profiles share the runtime profiles' original repository pins", () => {
    for (const target of ["linux", "windows"]) {
      const runtime = sdkComponents(target), compiler = sdkComponents("compiler");
      assert.deepEqual(Object.keys(compiler), [...Object.keys(runtime), "vcIntrinsics", "spirvHeaders"]);
      for (const name of Object.keys(runtime)) {assert.equal(Reflect.get(compiler, name).revision, Reflect.get(runtime, name).revision);}
    }
  });
  test("the shared compiler profile includes original LLVM, Clang and complete source-JIT bodies", () => {
    const c = sdkComponents("compiler").dpcpp;
    for (const name of ["llvm/lib", "llvm/include", "clang/lib", "clang/include", "clang/examples", "sycl-jit", "llvm-spirv", "libdevice"]) {assert.ok(c.paths.includes(name), name);}
    for (const name of ["llvm/CMakeLists.txt", "clang/CMakeLists.txt", "sycl/CMakeLists.txt", "LICENSE.TXT", "llvm/LICENSE.TXT", "clang/LICENSE.TXT", "sycl/LICENSE.TXT"]) {assert.ok(c.leaves.includes(name), name);}
  });
  test("compiler preparation applies only the reviewed complete source-JIT patch", () => {
    const c = sdkComponents("compiler"); assert.deepEqual(c.dpcpp.patches, ["dpcpp-static-jit.patch"]);
    for (const [name, value] of Object.entries(c)) {if (name !== "dpcpp") {assert.deepEqual(value.patches, []);}}
  });
  test("every complete JIT patch input belongs to the selected original compiler graph", () => {
    const c = sdkComponents("compiler").dpcpp, patch = fs.readFileSync(path.resolve(__dirname, "../runtime/dpcpp-static-jit.patch"), "utf8");
    for (const match of patch.matchAll(/^diff --git a\/(\S+) b\/\S+$/gm)) {
      const name = match[1] || ""; assert.ok(c.leaves.includes(name) || c.paths.some(root => name.startsWith(root + "/")), name);
    }
  });
  test("unused compiler tests, docs and symlinked ML training utilities stay outside the SDK graph", () => {
    const c = sdkComponents("compiler").dpcpp;
    for (const name of ["llvm", "clang", "llvm/test", "clang/test", "llvm/docs", "llvm/utils", "llvm/utils/mlgo-utils"]) {assert.ok(!c.paths.includes(name), name);}
  });
  test("compiler source selections cannot mutate the runtime profile", () => {
    const before = sdkComponents("windows"), c = sdkComponents("compiler"); c.dpcpp.paths.length = 0; c.dpcpp.patches.push("bad");
    assert.deepEqual(sdkComponents("windows"), before); assert.ok(sdkComponents("compiler").dpcpp.paths.includes("clang/lib"));
  });
  test("runtime source receipts cannot certify a compiler cohort", t => {
    const f = fixture(t), file = path.join(f.root, "receipt.json"); fs.writeFileSync(file, JSON.stringify({version: 2, target: "windows", status: "PASS"}));
    assert.throws(() => verifyPreparedSdkSources(file, "compiler"), /wrong-target/);
  });
  test("compiler receipts cannot be passed to a runtime SDK builder", t => {
    const f = fixture(t), file = path.join(f.root, "receipt.json"); fs.writeFileSync(file, JSON.stringify({version: 2, target: "compiler", status: "PASS"}));
    assert.throws(() => verifyPreparedSdkSources(file, "windows"), /wrong-target/);
  });
  test("compiler preparation retains explicit build-only admission before any output", t => {
    const f = fixture(t); assert.throws(() => prepareSdkSources("compiler", f.roots, f.output, {}), /MOM_AOT_BUILD_ONLY/); assert.ok(!fs.existsSync(f.output));
  });
  test("host-specific compiler aliases cannot duplicate the shared source profile", () => {
    for (const target of ["linux-compiler", "windows-compiler"]) {assert.throws(() => sdkComponents(target), /shared JIT sources/);}
  });
  test("compiler inputs require the original pinned GenX source instead of an implicit fetch", t => {
    const f = fixture(t), selected = sdkComponents("compiler"), vc = selected.vcIntrinsics; assert.ok(vc);
    assert.equal(vc.revision, "4fc83e12979096db72c129bd432238d5ca397e4d");
    assert.throws(() => validateSourceRoots(f.roots, f.output, selected), /all 10 pinned/);
    assert.equal(Object.keys(validateSourceRoots({...f.roots, vcIntrinsics: f.source, spirvHeaders: f.source}, f.output, selected)).length, 10);
  });
  test("SPIR-V translator headers use the original dependency pin and retain license metadata", () => {
    const selected = sdkComponents("compiler"), headers = Reflect.get(selected, "spirvHeaders"); assert.ok(headers);
    assert.equal(headers.revision, "5c50cbd25a40f8b60e44a2ccc2f1ba3c9e0d0299");
    assert.deepEqual(headers.paths, ["include", "LICENSES"]); assert.deepEqual(headers.leaves, ["LICENSE", "REUSE.toml"]);
    assert.deepEqual(headers.patches, []);
  });
});

test.describe("Unified SDK target and provenance boundaries", () => {
  test("Linux and Windows consume exactly the same original pinned repositories", () => {
    const linux = sdkComponents("linux"), windows = sdkComponents("windows");
    assert.deepEqual(Object.keys(linux), Object.keys(windows));
    for (const key of Object.keys(linux)) {
      const a = Reflect.get(linux, key), b = Reflect.get(windows, key);
      assert.equal(a.revision, b.revision); assert.deepEqual(a.paths, b.paths); assert.deepEqual(a.leaves, b.leaves);
    }
  });
  test("Linux keeps original host/adapter/JIT patches without Windows build adaptations", () => {
    const c = sdkComponents("linux");
    assert.deepEqual(c.dpcpp.patches, ["dpcpp-static-adapters.patch", "dpcpp-static-opencl.patch", "dpcpp-static-jit.patch"]);
    assert.deepEqual(c.umf.patches, ["umf-static-level-zero.patch"]); assert.deepEqual(c.openclLoader.patches, []);
    assert.deepEqual(c.levelZero.patches, ["level-zero-static-complete.patch"]);
  });
  test("Windows target keeps all prior reviewed patches in their original order", () => {
    assert.deepEqual(sdkComponents("windows").dpcpp.patches, components.dpcpp.patches);
    assert.equal(sdkComponents("windows").dpcpp.patches.length, 5);
  });
  test("a caller cannot mutate one target's patch policy through another returned plan", () => {
    const c = sdkComponents("linux"); c.dpcpp.patches.push("bad"); c.computeHeaders.paths.length = 0;
    assert.equal(sdkComponents("linux").dpcpp.patches.length, 3); assert.equal(sdkComponents("windows").computeHeaders.paths.length, 1);
  });
  test("unknown targets fail before reading roots, tools, or source receipts", () => {
    for (const target of ["", "win32", "Linux", "other"]) {
      assert.throws(() => sdkComponents(target), /linux or windows/);
      assert.throws(() => prepareSdkSources(target, {}, "/unavailable", {}), /linux or windows/);
      assert.throws(() => verifyPreparedSdkSources("/unavailable", target), /linux or windows/);
    }
  });
  test("a legacy unscoped receipt cannot be relabeled as a current target cohort", t => {
    const f = fixture(t), file = path.join(f.root, "receipt.json"); fs.writeFileSync(file, JSON.stringify({version: 1, target: "windows", status: "PASS"}));
    assert.throws(() => verifyPreparedSdkSources(file, "windows"), /wrong-target/);
  });
  test("a Windows source receipt cannot be consumed by the Linux build", t => {
    const f = fixture(t), file = path.join(f.root, "receipt.json"); fs.writeFileSync(file, JSON.stringify({version: 2, target: "windows", status: "PASS"}));
    assert.throws(() => verifyPreparedSdkSources(file, "linux"), /wrong-target/);
  });
  test("Linux production retains the same explicit build-only admission", t => {
    const f = fixture(t); assert.throws(() => prepareSdkSources("linux", f.roots, f.output, {}), /MOM_AOT_BUILD_ONLY/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("the actual CLI requires an explicit target instead of guessing from the build host", () => {
    const result = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/prepare-native-sdk-sources.js"), "linux"],
      {encoding: "utf8", timeout: 5000, env: {...process.env, PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /Usage:/); assert.equal(result.stdout, "");
  });
  test("every target's patch syntax remains valid before any original source is modified", () => {
    for (const target of ["linux", "windows"]) {
      for (const name of new Set(Object.values(sdkComponents(target)).flatMap(c => c.patches))) {
        const result = spawnSync("git", ["apply", "--numstat", path.resolve(__dirname, "../runtime", name)], {encoding: "utf8", timeout: 5000});
        assert.equal(result.status, 0, name + ": " + result.stderr);
      }
    }
  });
});

test.describe("Native Windows pinned SDK source preparation", () => {
  test("the eight upstream revisions reuse the existing reviewed source pins", () => {
    assert.equal(Object.keys(components).length, 8);
    assert.equal(components.dpcpp.revision, pins.source);
    for (const [key, component] of Object.entries(components)) {
      assert.match(component.revision, /^[0-9a-f]{40}$/);
      if (key !== "dpcpp") {assert.equal(component.revision, Reflect.get(pins, key));}
    }
  });
  test("the full native loader fixes are part of preparation, not a caller-supplied patch list", () => {
    assert.deepEqual(components.dpcpp.patches, ["dpcpp-static-adapters.patch", "dpcpp-static-windows.patch", "dpcpp-static-opencl.patch", "dpcpp-static-windows-cuda.patch", "dpcpp-static-jit.patch"]);
    assert.deepEqual(components.umf.patches, ["umf-static-level-zero.patch", "umf-static-windows.patch"]);
    assert.deepEqual(components.openclLoader.patches, ["opencl-static-windows.patch"]);
  });
  test("the experimental header provider exports only required headers and its license", () => {
    assert.deepEqual(components.computeHeaders.paths, ["level_zero/include"]);
    assert.deepEqual(components.computeHeaders.leaves, ["LICENSE.md"]);
  });
  test("runtime source-JIT uses the original public ABI headers and scoped host patch", () => {
    assert.ok(components.dpcpp.paths.includes("sycl-jit/jit-compiler/include"));
    const patch = path.resolve(__dirname, "../runtime/dpcpp-static-jit.patch");
    for (const target of ["linux", "windows"]) {
      assert.deepEqual(sdkPatchScope(target, patch), ["--include=sycl/*"]);
      assert.deepEqual(sdkPatchScope("compiler", patch), []);
      assert.deepEqual(sdkPatchScope(target, path.resolve(__dirname, "../runtime/dpcpp-static-adapters.patch")), []);
    }
  });
  test("all reviewed source patches have valid unified-diff boundaries", () => {
    for (const name of new Set(Object.values(components).flatMap(component => component.patches))) {
      const run = spawnSync("git", ["apply", "--numstat", path.join(__dirname, "../runtime", name)], {encoding: "utf8", timeout: 5000});
      assert.equal(run.status, 0, name + ": " + run.stderr);
    }
  });
  test("only a complete fixed-key source-root descriptor is admitted", t => {
    const f = fixture(t);
    assert.deepEqual(validateSourceRoots(f.roots, f.output), f.roots);
    for (const input of [null, [], "", {}, {...f.roots, extra: f.source}, {...f.roots, umf: undefined}]) {
      assert.throws(() => validateSourceRoots(input, f.output));
    }
    assert.equal(fs.existsSync(f.output), false);
  });
  test("an existing output is never reused or deleted", t => {
    const f = fixture(t); fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "evidence"), "keep");
    assert.throws(() => validateSourceRoots(f.roots, f.output), /new absolute output/);
    assert.equal(fs.readFileSync(path.join(f.output, "evidence"), "utf8"), "keep");
  });
  test("outputs inside source repositories are rejected before any mutation", t => {
    const f = fixture(t); const output = path.join(f.source, "build");
    assert.throws(() => validateSourceRoots(f.roots, output), /overlaps/);
    assert.equal(fs.existsSync(output), false);
  });
  test("symlink aliases cannot evade root or output overlap checks", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), alias = path.join(f.root, "alias"); fs.symlinkSync(f.source, alias, "dir");
    assert.throws(() => validateSourceRoots({...f.roots, umf: alias}, f.output), /real directory/);
    assert.throws(() => validateSourceRoots(f.roots, path.join(alias, "child")), /overlaps/);
  });
  test("source inventories hash actual bytes and preserve deterministic membership", t => {
    const f = fixture(t); fs.mkdirSync(path.join(f.source, "nested"));
    fs.writeFileSync(path.join(f.source, "z.h"), "before"); fs.writeFileSync(path.join(f.source, "nested/a.h"), "nested");
    const before = sdkSourceFiles(f.source);
    assert.deepEqual(Object.keys(before), ["nested/a.h", "z.h"]); assert.equal(before["z.h"], fileHash(path.join(f.source, "z.h")));
    fs.writeFileSync(path.join(f.source, "z.h"), "after"); assert.notDeepEqual(sdkSourceFiles(f.source), before);
  });
  test("export verification matches Git's actual blob identity and rejects substituted bytes", t => {
    const f = fixture(t), file = path.join(f.source, "a.h"); fs.writeFileSync(file, "original\n");
    const result = spawnSync("git", ["hash-object", file], {encoding: "utf8", timeout: 5000}); assert.equal(result.status, 0, result.stderr);
    const tree = "100644 blob " + result.stdout.trim() + "\ta.h\0", files = sdkSourceFiles(f.source);
    verifySdkExport(f.source, files, tree);
    fs.writeFileSync(file, "substitute\n"); assert.throws(() => verifySdkExport(f.source, files, tree), /pinned blob/);
  });
  test("Git archive exclusions or extra exported files cannot certify a source cohort", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.source, "extra.h"), "extra");
    assert.throws(() => verifySdkExport(f.source, sdkSourceFiles(f.source), ""), /outside its pinned tree/);
    assert.throws(() => verifySdkExport(f.source, {}, "100644 blob " + "a".repeat(40) + "\tmissing.h\0"), /membership/);
    assert.throws(() => verifySdkExport(f.source, {}, "120000 blob " + "a".repeat(40) + "\tescape\0"), /membership/);
  });
  test("source inventories cannot follow an untracked include link outside the SDK", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.symlinkSync(f.root, path.join(f.source, "escape"), "dir");
    assert.throws(() => sdkSourceFiles(f.source), /bounded regular files/);
  });
  test("a sparse oversized source is rejected before attempting to hash its contents", t => {
    const f = fixture(t), file = path.join(f.source, "large"); fs.writeFileSync(file, ""); fs.truncateSync(file, (1 << 30) + 1);
    assert.throws(() => sdkSourceFiles(f.source), /bounded regular files/);
  });
  test("parent repository redirects and extraction hooks cannot enter source preparation", () => {
    const input = {PATH: "/bin", GIT_DIR: "/unrelated", GIT_WORK_TREE: "/unrelated", GIT_CONFIG_COUNT: "1", GIT_NO_REPLACE_OBJECTS: "0", TAR_OPTIONS: "--to-command=unrelated"};
    const env = sdkSourceEnvironment(input);
    for (const key of ["GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_COUNT", "TAR_OPTIONS"]) {assert.equal(env[key], undefined);}
    assert.equal(env.GIT_NO_REPLACE_OBJECTS, "1"); assert.equal(input.GIT_NO_REPLACE_OBJECTS, "0"); assert.equal(env.GIT_NO_LAZY_FETCH, "1");
  });
  test("preparation needs explicit build-only admission before examining repositories", t => {
    const f = fixture(t); assert.throws(() => prepareSdkSources("windows", f.roots, f.output, {}), /MOM_AOT_BUILD_ONLY/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("missing pinned Git objects cannot create a purported source cohort", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t); const initialized = spawnSync("git", ["init", "--quiet", f.source], {encoding: "utf8", timeout: 5000}); assert.equal(initialized.status, 0, initialized.stderr);
    assert.throws(() => prepareSdkSources("windows", f.roots, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}),
      /source input query failed: dpcpp[\s\S]*source input query failed: emhash[\s\S]*source input query failed: openclHeaders/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("Windows static UMF opts into one CRT and excludes only native archive resources", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/umf-static-windows.patch"), "utf8");
    const cmake = fs.readFileSync(path.join(__dirname, "native/dpcpp-static/CMakeLists.txt"), "utf8");
    assert.match(patch, /if\(MOM_STATIC_WINDOWS\)[\s\S]*?\/MT/); assert.match(patch, /if\(NOT MOM_STATIC_WINDOWS\)/);
    assert.match(cmake, /set\(MOM_STATIC_WINDOWS ON CACHE BOOL/); assert.match(cmake, /if\(umf_static_crt EQUAL -1\)/);
    assert.match(patch, /git apply --reverse --check \$\{PROJECT_SOURCE_DIR\}\/cmake\/fix_coverity_issues.patch/);
    assert.match(patch, /if\(NOT UMF_HWLOC_VERIFY_RESULT EQUAL 0\)/);
  });
  test("header-only Windows CUDA adapter builds do not need a compiler version-query stub", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/dpcpp-static-windows-cuda.patch"), "utf8");
    assert.equal(patch.match(/if\(NOT \(WIN32 AND UR_STATIC_ADAPTER_CUDA AND MOM_STATIC_WINDOWS\)\)/g)?.length, 2);
    const cmake = fs.readFileSync(path.join(__dirname, "native/dpcpp-static/CMakeLists.txt"), "utf8");
    assert.match(cmake, /CUDAToolkit_INCLUDE_DIRS "\$\{MOM_DPCPP_CUDA_SDK\}\/include"/);
    assert.match(cmake, /CMAKE_MATCH_1 LESS 10010/);
  });
  test("embedded Level Zero keeps vendor diagnostics without promoting them to Clang errors", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/level-zero-static-windows.patch"), "utf8");
    assert.match(patch, /MOM_EMBEDDED_LEVEL_ZERO AND CMAKE_CXX_COMPILER_ID STREQUAL Clang/);
    assert.doesNotMatch(patch, /\+.*(?:-w|Wno-|WX-)/);
    assert.ok(components.levelZero.patches.includes("level-zero-static-windows.patch"));
  });
  test("Windows raw adapter targets cannot dispatch the ELF isolation helpers", () => {
    const cmake = fs.readFileSync(path.join(__dirname, "native/dpcpp-static/CMakeLists.txt"), "utf8");
    for (const option of ["MOM_DPCPP_LEVEL_ZERO_SOURCE", "MOM_DPCPP_CUDA_SDK"]) {
      assert.ok(cmake.includes(`if(${option} AND NOT WIN32)\n  find_program(MOM_NODE node REQUIRED)`));
    }
    assert.match(cmake, /elseif\(MOM_DPCPP_LEVEL_ZERO_SOURCE\)[\s\S]*?INTERFACE ur_adapter_level_zero_v2/);
    assert.match(cmake, /elseif\(MOM_DPCPP_CUDA_SDK\)[\s\S]*?INTERFACE ur_adapter_cuda/);
    assert.ok(cmake.includes('target_link_options(LevelZeroLoader INTERFACE "/wholearchive:$<TARGET_FILE:ze_loader>")'));
  });
});
