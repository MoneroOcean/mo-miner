"use strict";
const fs = require("node:fs"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {linuxStaticSdkPlan, linuxSdkJitInput, linuxSdkHeaders, linuxSdkDependencies, buildLinuxStaticSdk} = require("../scripts/build-native-dpcpp-sdk");
const {fileHash} = require("../scripts/build-source-inputs"), {dpcppEntries} = require("../scripts/native-compiler-isolation");
const {validateSourceRoots} = require("../scripts/prepare-native-sdk-sources");
const {tempDir} = require("./common/temp-dir");
const roots = Object.fromEntries(["dpcpp", "emhash", "umf", "levelZero", "computeHeaders", "hwloc", "openclLoader", "openclHeaders"].map(n => [n, "/source/" + n]));
const tools = {cmake: "/tools/cmake", ninja: "/tools/ninja"};
const plan = () => linuxStaticSdkPlan(roots, "/sdk/dpcpp", "/sdk/cuda", "/jit/compiler.o", "/new/linux-sdk", tools, 2);
/** @param {import("node:test").TestContext} t */
function jit(t) {
  const root = tempDir(t, "mom-dpcpp-sdk-");
  const object = path.join(root, "compiler.o"); fs.writeFileSync(object, "synthetic isolated ELF object");
  const receipt = {prefix: "mom_dpcpp23_", entryPoints: dpcppEntries, objectSha256: fileHash(object)};
  const save = () => fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(receipt)); save(); return {root, object, receipt, save};
}

test.describe("Original Linux static SYCL SDK production", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("source-JIT isolation keeps the binding fixture's complete bridge cohort", () => {
    const source = fs.readFileSync(path.resolve(__dirname, "native/dpcpp-jit-binding.cpp"), "utf8");
    /** @param {string} text */
    const entries = (text) => [...(text.match(/#define ENTRY\(name\)([\s\S]*?)#undef ENTRY/)?.[1] || "").matchAll(/\bENTRY\(([A-Za-z_]\w*)\)/g)].map(match => match[1] || "");
    /** @param {string} text */
    const verify = (text) => assert.deepEqual(entries(text), dpcppEntries);
    verify(source);
    const changed = source.replace("ENTRY(destroyBinary)", "ENTRY(destroyBinary) ENTRY(newBridge)");
    assert.notEqual(changed, source); assert.throws(() => verify(changed));
  });

  test("the producer reuses the existing CMake graph rather than another runtime registry", () => {
    const p = plan(); assert.ok(p.configure.includes(path.resolve(__dirname, "native/dpcpp-static")));
    assert.ok(p.compile.includes("native-dpcpp-core")); assert.deepEqual(p.compile.slice(-5), ["--parallel", "2", "--", "-k", "0"]);
  });
  test("all adapter sources, loaders, headers and source JIT are explicit", () => {
    const p = plan(); for (const key of ["dpcpp", "emhash", "umf", "levelZero", "computeHeaders", "openclLoader", "openclHeaders"]) {
      assert.ok(p.configure.some(arg => arg.endsWith("=" + roots[key])), key);
    }
    assert.ok(p.configure.includes("-DMOM_DPCPP_JIT_OBJECT=/jit/compiler.o"));
    assert.ok(p.configure.includes("-DMOM_DPCPP_CUDA_SDK=/sdk/cuda"));
  });
  test("mutable autotools hwloc configuration is confined to a private output snapshot", () => {
    assert.ok(plan().configure.includes("-DMOM_DPCPP_HWLOC=/new/linux-sdk/hwloc"));
    assert.ok(!plan().configure.some(arg => arg.endsWith("=" + roots["hwloc"])));
  });
  test("the compiler and build runner are selected literally, never through a shell", () => {
    const p = plan(); assert.ok(p.configure.includes("-DCMAKE_CXX_COMPILER=/sdk/dpcpp/bin/clang++"));
    assert.ok(p.configure.includes("-DCMAKE_MAKE_PROGRAM=/tools/ninja")); assert.equal(p.configure[0], tools.cmake);
  });
  test("explicit host compilers do not replace the pinned device/JIT SDK", () => {
    const p = linuxStaticSdkPlan(roots, "/sdk/dpcpp", "/sdk/cuda", "/jit/compiler.o", "/new/linux-sdk",
      {...tools, cc: "/intel/icx", cxx: "/intel/icpx"}, 2);
    assert.ok(p.configure.includes("-DCMAKE_C_COMPILER=/intel/icx"));
    assert.ok(p.configure.includes("-DCMAKE_CXX_COMPILER=/intel/icpx"));
    assert.ok(p.configure.includes("-DMOM_DPCPP_SDK=/sdk/dpcpp"));
    assert.deepEqual(p.hostCompilers, {cc: "/intel/icx", cxx: "/intel/icpx"});
  });
  test("host compiler overrides must be paired absolute literals", () => {
    for (const selected of [{...tools, cc: "/intel/icx"}, {...tools, cxx: "/intel/icpx"},
      {...tools, cc: "icx", cxx: "/intel/icpx"}, {...tools, cc: "/intel/icx;other", cxx: "/intel/icpx"}]) {
      assert.throws(() => linuxStaticSdkPlan(roots, "/sdk", "/cuda", "/jit/compiler.o", "/new", selected, 2));
    }
  });
  test("host compiler inputs cannot be inside the output tree", () => {
    assert.throws(() => linuxStaticSdkPlan(roots, "/sdk", "/cuda", "/jit/compiler.o", "/new",
      {...tools, cc: "/new/icx", cxx: "/intel/icpx"}, 2), /overlaps/);
  });
  test("nested hwloc Autoconf uses shared safe names while CMake keeps absolute compilers", () => {
    const cc = "/Intel install/icx", cxx = "/Intel's install/icpx";
    const p = linuxStaticSdkPlan(roots, "/sdk", "/cuda", "/jit/compiler.o", "/new", {...tools, cc, cxx}, 2);
    assert.deepEqual(p.compilerEnvironment, {CC: "icx", CXX: "icpx"});
    assert.ok(p.configure.includes("-DCMAKE_C_COMPILER=" + cc)); assert.ok(p.configure.includes("-DCMAKE_CXX_COMPILER=" + cxx));
    assert.deepEqual(plan().compilerEnvironment, {CC: "clang", CXX: "clang++"});
  });
  test("DPC++ producer recipe binds the shared host compiler environment", () => {
    const script = fs.readFileSync(path.resolve(__dirname, "../scripts/build-native-dpcpp-sdk.js"), "utf8");
    for (const file of ["scripts/build-native-sycl-sdk.js", "scripts/build-native-release.js", "scripts/native-sycl-inputs.js",
      "scripts/native-jit-link.js", "scripts/native-comgr-inputs.js"]) {assert.ok(script.includes('"' + file + '"'), file);}
  });
  test("relative roots, CMake interpolation, missing profiles and invalid job counts fail closed", () => {
    for (const jobs of [0, 17, NaN, 1.5]) {assert.throws(() => linuxStaticSdkPlan(roots, "/sdk", "/cuda", "/jit/compiler.o", "/new", tools, jobs), /explicit tools\/JIT/);}
    for (const root of ["relative", "/sdk;other", "/sdk$ENV{PATH}"]) {assert.throws(() => linuxStaticSdkPlan(roots, root, "/cuda", "/jit/compiler.o", "/new", tools, 2), /explicit tools\/JIT/);}
    const missing = {...roots}; delete missing["umf"]; assert.throws(() => linuxStaticSdkPlan(missing, "/sdk", "/cuda", "/jit/compiler.o", "/new", tools, 2));
  });
  test("output cannot replace, contain or enter any selected input", () => {
    for (const output of ["/source", "/source/umf", "/source/umf/build", "/jit", "/sdk/dpcpp/new"]) {
      assert.throws(() => linuxStaticSdkPlan(roots, "/sdk/dpcpp", "/sdk/cuda", "/jit/compiler.o", output, tools, 2), /overlaps/);
    }
  });
  test("SDK compilation cannot start without the explicit build-only boundary", () => {
    assert.throws(() => buildLinuxStaticSdk("/missing", "/sdk", "/cuda", "/jit/compiler.o", "/new", tools, 2, {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("shared realpath admission catches output aliases into original source roots", {skip: process.platform === "win32"}, t => {
    const f = jit(t), source = path.join(f.root, "source"), alias = path.join(f.root, "alias"); fs.mkdirSync(source); fs.symlinkSync(source, alias);
    const selected = Object.fromEntries(Object.keys(roots).map(key => [key, source]));
    assert.throws(() => validateSourceRoots(selected, path.join(alias, "build")), /overlaps/);
    assert.equal(fs.existsSync(path.join(source, "build")), false);
  });
  test("isolated source-JIT bytes and its complete original entrypoint cohort are bound", t => {
    const f = jit(t), result = linuxSdkJitInput(f.object); assert.equal(result[f.object], f.receipt.objectSha256);
    assert.equal(Object.keys(result).length, 2);
  });
  test("changed source-JIT bytes cannot be certified by the old isolation receipt", t => {
    const f = jit(t); fs.appendFileSync(f.object, "changed"); assert.throws(() => linuxSdkJitInput(f.object), /isolation receipt/);
  });
  test("a reduced source-JIT interface cannot silently disable an original entrypoint", t => {
    const f = jit(t); f.receipt.entryPoints = f.receipt.entryPoints.slice(1); f.save(); assert.throws(() => linuxSdkJitInput(f.object), /isolation receipt/);
  });
  test("stale or unknown Ninja records cannot certify compiler dependencies", () => {
    for (const text of ["", "a.o: #deps 1, deps mtime 1 (STALE)\n    /a.h\n", "a.o: #deps 1, deps mtime 1 (VALID)\nnot-a-header\n", "    /a.h\n"]) {
      assert.throws(() => linuxSdkDependencies(text, "/build"), /SDK dependenc/);
    }
  });
  test("Ninja dependency paths preserve spaces, resolve relative entries and deduplicate", () => {
    const s = "first.o: #deps 2, deps mtime 123 (VALID)\n    /source/a b.h\n    generated.h\n\nsecond.o: #deps 1, deps mtime 456 (VALID)\n    /source/a b.h\n";
    assert.deepEqual(linuxSdkDependencies(s, "/build"), ["/source/a b.h", "/build/generated.h"]);
  });
  test("control characters in Ninja inputs are rejected before filesystem access", () => {
    assert.throws(() => linuxSdkDependencies("a.o: #deps 1, deps mtime 1 (VALID)\n    /a\x00b.h\n", "/build"), /SDK dependency record/);
  });
  test("missing original NVML headers fail admission before compiler configuration", t => {
    const f = jit(t), sdk = path.join(f.root, "sdk"), cuda = path.join(f.root, "cuda");
    fs.mkdirSync(path.join(sdk, "include/sycl"), {recursive: true}); fs.mkdirSync(path.join(cuda, "include"), {recursive: true});
    fs.writeFileSync(path.join(sdk, "include/sycl/sycl.hpp"), "sycl"); fs.writeFileSync(path.join(cuda, "include/cuda.h"), "cuda");
    assert.throws(() => linuxSdkHeaders(sdk, cuda), /original header: .*nvml\.h/);
    fs.writeFileSync(path.join(cuda, "include/nvml.h"), "nvml"); assert.equal(linuxSdkHeaders(sdk, cuda).length, 2);
  });
});
