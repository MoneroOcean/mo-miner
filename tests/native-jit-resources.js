"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs");
const {prepare, validateProfile, resourceTarget, verifyJitResources, resourceSourceInputs} = require("../scripts/prepare-native-sycl-jit-resources");
const {jitEmitterPlan, buildJitEmitter, jitSourceInputs} = require("../scripts/native-jit-sdk");
const producerSource = fs.readFileSync(path.join(__dirname, "../scripts/prepare-native-sycl-jit-resources.js"), "utf8");
const buildOnlyInput = "tests/native/dpcpp-jit/CMakeLists.txt";

/** @typedef {{version: number, target: string, fixture?: string, clang?: string, receipt?: string, gnuRoot?: string, baseRoot?: string,
 * noticeRoot?: string, syclInclude?: string, clangBuiltinInclude?: string, baseManifest?: string, headerRoots?: Array<{path: string, virtual: string}>,
 * required?: {spirv32: {virtual: string, sha256: string}, spirv64: {virtual: string, sha256: string}, libdevice: {virtual: string, sha256: string}},
 * notices?: Array<{source: string, virtual: string, sha256: string}>}} TestProfile */

/** @param {import("node:test").TestContext} t */
function directory(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sycl-jit-resources-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true})); return root;
}

/** @param {string} name @returns {string | undefined} */
function findClang(name = "clang++") {
  const candidates = process.env["CXX"] && path.isAbsolute(process.env["CXX"]) ? [process.env["CXX"]] : ["/opt/dpcpp/bin/clang++", name];
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["--version"], {encoding: "utf8", timeout: 5000});
    if (result.status !== 0 || result.error) {continue;}
    const executable = path.isAbsolute(candidate) ? candidate : (() => {
      const found = spawnSync(process.platform === "win32" ? "where.exe" : "which", [candidate], {encoding: "utf8", timeout: 5000});
      return found.status === 0 ? found.stdout.trim().split(/\r?\n/)[0] : undefined;
    })();
    if (!executable) {continue;}
    const probe = spawnSync(executable, ["--target=x86_64-w64-windows-gnu", "-fsycl", "-fsycl-device-only", "-nostdinc", "-nostdinc++", "-x", "c++", "-fsyntax-only", "-"], {
      input: "int mom_sycl_probe() { return 17; }\n", encoding: "utf8", timeout: 10000,
    });
    if (probe.status === 0 && !probe.error) {return executable;}
  }
  return undefined;
}

/** @param {string} file @param {string | Buffer} data */
function write(file, data) {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data, {flag: "wx"});}

/** @param {string} root @param {string} source @param {string} virtual */
function baseEntry(root, source, virtual) {
  const file = path.join(root, source); return {source, virtual, sha256: fileHash(file)};
}

/** @param {import("node:test").TestContext} t @param {string} [target] @returns {TestProfile} */
function actualProfile(t, target = "x86_64-w64-windows-gnu") {
  const clang = findClang();
  if (!clang) {t.skip("clang++ unavailable for the bounded producer proof"); return /** @type {TestProfile} */ ({});}
  const root = directory(t), gnuRoot = path.join(root, "gnu-package"), baseRoot = path.join(root, "base-sdk"), noticeRoot = path.join(root, "notices"), build = path.join(root, "build");
  const selected = resourceTarget(target), common = selected.roots.at(-1) + "/common.h";
  for (const directoryName of [...selected.roots, "include/mom-cxx/backward", "usr/share/licenses/mom"]) {
    fs.mkdirSync(path.join(gnuRoot, directoryName), {recursive: true});
  }
  for (const directoryName of ["include/sycl", "lib/clang/23/include"]) {
    fs.mkdirSync(path.join(baseRoot, directoryName), {recursive: true});
  }
  fs.mkdirSync(noticeRoot, {recursive: true});
  write(path.join(gnuRoot, "include/mom-cxx/mom.h"), "#define MOM_HEADER 1\n");
  write(path.join(gnuRoot, "include/mom-cxx/backward/backward.h"), "#define MOM_BACKWARD 1\n");
  write(path.join(gnuRoot, "include/mom-cxx-target/target.h"), "#define MOM_TARGET 1\n");
  write(path.join(gnuRoot, "include/mom-system-target/system.h"), "#define MOM_SYSTEM 1\n");
  write(path.join(gnuRoot, common), "#define MOM_COMMON 1\n");
  const bitcode = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.alloc(12)]);
  write(path.join(baseRoot, "lib/spirv32.bc"), bitcode);
  write(path.join(baseRoot, "lib/spirv64.bc"), bitcode);
  write(path.join(baseRoot, "lib/libdevice.bc"), bitcode);
  write(path.join(noticeRoot, "LICENSE"), "test notice\n");
  const source = path.join(root, "fixture.cpp"), fixture = path.join(build, process.platform === "win32" ? "native-dpcpp-jit.exe" : "native-dpcpp-jit");
  write(source, ["#include <cstdio>", "#include <cstring>", "int main(int argc, char **argv) {",
    "  if (argc != 3 || std::strcmp(argv[1], \"--emit-source\") != 0) return 2;",
    "  std::printf(\"#include <mom.h>\\n#include <backward.h>\\n#include <common.h>\\nint mom_%s() { return MOM_HEADER + MOM_BACKWARD + MOM_COMMON; }\\n\", argv[2]); return 0;", "}", ""].join("\n"));
  fs.mkdirSync(build);
  const built = spawnSync(clang, ["-std=c++17", "-O0", source, "-o", fixture], {encoding: "utf8", timeout: 30000});
  if (built.status !== 0) {t.skip("clang++ cannot build the CPU-only fixture: " + (built.stderr || "")); return /** @type {TestProfile} */ ({});}
  const manifest = path.join(root, "base-manifest.json");
  const files = [baseEntry(baseRoot, "lib/spirv32.bc", "lib/spirv32.bc"), baseEntry(baseRoot, "lib/spirv64.bc", "lib/spirv64.bc"),
    baseEntry(baseRoot, "lib/libdevice.bc", "lib/libdevice.bc")];
  write(manifest, JSON.stringify({version: 1, files}, null, 2) + "\n");
  const receipt = path.join(root, "sdk-receipt.json"), sourceInputs = /** @type {Record<string, string>} */ ({});
  for (const relative of ["tests/reference/native-dpcpp-jit.cpp", "tests/reference/native-kawpow-source.cpp", "tests/native/dpcpp-jit/CMakeLists.txt",
    "scripts/native-resources.js", "sycl/kawpow/device.inc", "sycl/kawpow/epoch.h", "sycl/kawpow/jit.inc", "sycl/kawpow/keccak.inc",
    "sycl/kawpow/program.inc", "sycl/device-state.h", "sycl/lib-internal.h", "sycl/lib.h", "sycl/workgroup-limits.h",
    "xmrig/3rdparty/libethash/data_sizes.h", "xmrig/3rdparty/libethash/ethash.h"]) {
    const file = path.resolve(__dirname, "..", relative); sourceInputs[file] = fileHash(file);
  }
  for (const relative of ["include/mom-cxx/mom.h", "include/mom-cxx/backward/backward.h", "include/mom-cxx-target/target.h", "include/mom-system-target/system.h", common]) {
    const file = path.join(gnuRoot, relative); sourceInputs[file] = fileHash(file);
  }
  const receiptValue = {version: 1, fixture, fixtureSha256: fileHash(fixture), compiler: clang, compilerSha256: fileHash(clang), inputs: sourceInputs};
  write(receipt, JSON.stringify(receiptValue, null, 2) + "\n");
  return /** @type {TestProfile} */ ({version: 1, target, fixture, clang, receipt, gnuRoot, baseRoot,
    noticeRoot, syclInclude: path.join(baseRoot, "include/sycl"), clangBuiltinInclude: path.join(baseRoot, "lib/clang/23/include"), baseManifest: manifest,
    headerRoots: selected.roots.map((virtual) => ({path: path.join(gnuRoot, virtual), virtual})),
    required: {spirv32: {virtual: "lib/spirv32.bc", sha256: fileHash(path.join(baseRoot, "lib/spirv32.bc"))},
      spirv64: {virtual: "lib/spirv64.bc", sha256: fileHash(path.join(baseRoot, "lib/spirv64.bc"))},
      libdevice: {virtual: "lib/libdevice.bc", sha256: fileHash(path.join(baseRoot, "lib/libdevice.bc"))}},
    notices: [{source: "LICENSE", virtual: "usr/share/licenses/mom/LICENSE", sha256: fileHash(path.join(noticeRoot, "LICENSE"))}]});
}

test.describe("Compiler SDK resource admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @param {import('node:test').TestContext} t @param {string} [target] */
  function artifact(t, target = "x86_64-w64-windows-gnu") {
    const p = actualProfile(t, target); if (!p.fixture) {return null;}
    const root = path.join(path.dirname(p.fixture), "admission"); prepare(p, root);
    const file = path.join(root, "manifest.json"), manifest = JSON.parse(fs.readFileSync(file, "utf8"));
    return {root, file, manifest, save: () => fs.writeFileSync(file, JSON.stringify(manifest))};
  }
  for (const target of ["x86_64-w64-windows-gnu", "x86_64-unknown-linux-gnu"]) {
    test("admits complete current " + target + " resources without executing them", t => {
      const a = artifact(t, target); if (!a) {return;}
      const r = verifyJitResources(a.root, target); assert.equal(r.target, target); assert.equal(r.manifestHash, fileHash(a.file));
      assert.equal(r.inputs[path.join(a.root, "overlay/lib/libdevice.bc")], fileHash(path.join(a.root, "overlay/lib/libdevice.bc")));
    });
  }
  test("another target cannot consume the artifact", t => {
    const a = artifact(t); if (!a) {return;}
    assert.throws(() => verifyJitResources(a.root, "x86_64-unknown-linux-gnu"), /Wrong-target/);
  });
  test("current source fingerprints cannot be omitted or restamped as old inputs", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.sourceInputs[jitSourceInputs[0] || ""] = "0".repeat(64); a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /source is stale/);
  });
  test("the source footprint cannot contain unexpected fields", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.sourceInputs.extra = "a".repeat(64); a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /source footprint/);
  });
  test("accepts only the current resource source footprint", t => {
    const a = artifact(t); if (!a) {return;}
    assert.deepEqual(Object.keys(a.manifest.sourceInputs).sort(), [...resourceSourceInputs].sort());
    a.manifest.sourceInputs[buildOnlyInput] = "a".repeat(64); a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /source footprint/);
    delete a.manifest.sourceInputs[buildOnlyInput];
    delete a.manifest.sourceInputs[resourceSourceInputs[0] || ""]; a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /source footprint/);
  });
  test("payload traversal is rejected before following the path", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.files[0].virtual = "../outside"; a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /relative resource payload/);
  });
  test("altered payload bytes cannot match their old receipt", t => {
    const a = artifact(t); if (!a) {return;}
    fs.appendFileSync(path.join(a.root, "overlay", a.manifest.files[0].virtual), "changed");
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /size changed/);
  });
  test("undeclared payload files invalidate membership", t => {
    const a = artifact(t); if (!a) {return;}
    write(path.join(a.root, "overlay/extra.h"), "extra");
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /membership/);
  });
  test("case-folded names cannot alias one another", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.files.push({...a.manifest.files[0], virtual: a.manifest.files[0].virtual.toUpperCase()}); a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /duplicate/);
  });
  test("required LLVM libraries need bitcode magic as well as matching hashes", t => {
    const a = artifact(t); if (!a) {return;}
    const p = path.join(a.root, "overlay/lib/libdevice.bc"); fs.writeFileSync(p, Buffer.alloc(16));
    const hash = fileHash(p); a.manifest.required.libdevice.sha256 = hash;
    a.manifest.files.find((/** @type {{virtual: string}} */ row) => row.virtual === "lib/libdevice.bc").sha256 = hash; a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /not LLVM bitcode/);
  });
  test("required bitcode hashes retain the producer's case-insensitive semantics", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.required.libdevice.sha256 = a.manifest.required.libdevice.sha256.toUpperCase(); a.save();
    assert.equal(verifyJitResources(a.root, "x86_64-w64-windows-gnu").target, "x86_64-w64-windows-gnu");
  });
  test("all maintained generated variants must remain present and bounded", t => {
    const a = artifact(t); if (!a) {return;}
    a.manifest.generated[0].bytes = 0; a.save();
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /generated JIT source/);
  });
  test("notices participate in the payload integrity check", t => {
    const a = artifact(t); if (!a) {return;}
    fs.appendFileSync(path.join(a.root, "notices", a.manifest.notices[0].virtual), "changed");
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /size changed/);
  });
  test("resource symlinks cannot redirect admission outside the artifact", {skip: process.platform === "win32"}, t => {
    const a = artifact(t); if (!a) {return;}
    const p = path.join(a.root, "overlay", a.manifest.files[0].virtual), outside = path.join(directory(t), "outside");
    fs.copyFileSync(p, outside); fs.unlinkSync(p); fs.symlinkSync(outside, p);
    assert.throws(() => verifyJitResources(a.root, "x86_64-w64-windows-gnu"), /not regular/);
  });
});

test.describe("CPU-only JIT resource bootstrap", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("plans original source generation without linking a compiler SDK", () => {
    const p = jitEmitterPlan("/sdk", "/output");
    assert.equal(p.cxx, "/sdk/bin/clang++"); assert.equal(p.binary, "/output/native-kawpow-source");
    assert.ok(p.compile.includes("-DMOM_KAWPOW_SOURCE_EMITTER"));
    assert.equal(p.compile.at(-3), path.resolve(__dirname, "reference/native-kawpow-source.cpp"));
    assert.ok(!p.compile.some(value => /\.a$|\.lib$|\.so$|^-lsycl$|^-fsycl$/.test(value)));
  });
  test("uses the same embedded mining source bytes and public SDK type declarations", () => {
    const p = jitEmitterPlan("/sdk space", "/output space");
    assert.deepEqual(p.resources.slice(3).filter((_, i) => i % 2 === 0), ["kawpow-device.inc", "kawpow-keccak.inc"]);
    assert.ok(p.compile.includes("-I/sdk space/include")); assert.ok(p.compile.includes("-DMOM_EMBEDDED_SYCL_JIT"));
  });
  test("resource footprint excludes build-only CMake input while full producer recipe retains it", () => {
    assert.ok(Object.isFrozen(jitSourceInputs)); assert.ok(Object.isFrozen(resourceSourceInputs));
    assert.ok(jitSourceInputs.includes("sycl/kawpow/program.inc"));
    assert.ok(jitSourceInputs.includes("tests/reference/native-kawpow-source.cpp"));
    assert.ok(jitSourceInputs.includes(buildOnlyInput)); assert.ok(!resourceSourceInputs.includes(buildOnlyInput));
    assert.deepEqual(resourceSourceInputs, jitSourceInputs.filter(name => name !== buildOnlyInput));
  });
  test("does not mutate earlier or later plans", () => {
    const p = jitEmitterPlan("/sdk", "/output"); p.compile.length = 0;
    assert.ok(jitEmitterPlan("/sdk", "/output").compile.includes("-std=c++17"));
  });
  test("refuses nonliteral SDK/output values before executing tools", () => {
    for (const value of ["relative", "/a;bad", "/a\nbad", "/a$bad", "/a/../b"]) {assert.throws(() => jitEmitterPlan(value, "/output")); assert.throws(() => jitEmitterPlan("/sdk", value));}
  });
  test("refuses output inside or around source and installed SDK roots", () => {
    for (const output of ["/sdk", "/sdk/output", "/", path.resolve(__dirname, "..")]) {assert.throws(() => jitEmitterPlan("/sdk", output), /overlaps/);}
  });
  test("requires explicit build-only admission before reading a missing SDK", t => {
    const root = directory(t), output = path.join(root, "output");
    assert.throws(() => buildJitEmitter("/missing-sdk", output, {}), /MOM_AOT_BUILD_ONLY/); assert.ok(!fs.existsSync(output));
  });
  test("target-header input lists cannot add relative, excessive or interpolated paths", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const output = path.join(directory(t), "output");
    for (const headers of [null, {}, ["relative"], ["/a;bad"], Array(16385).fill("/header")]) {
      assert.throws(() => buildJitEmitter("/missing-sdk", output, {MOM_AOT_BUILD_ONLY: "1"}, headers), /bounded literal/);
    }
    assert.ok(!fs.existsSync(output));
  });
  test("never overwrites an existing output or its evidence", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const root = directory(t), output = path.join(root, "output"); fs.mkdirSync(output); write(path.join(output, "keep"), "evidence");
    assert.throws(() => buildJitEmitter("/missing-sdk", output, {MOM_AOT_BUILD_ONLY: "1"}), /existing evidence/);
    assert.equal(fs.readFileSync(path.join(output, "keep"), "utf8"), "evidence");
  });
  test("missing SDK headers fail before creating output", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const root = directory(t), sdk = path.join(root, "sdk"), output = path.join(root, "output"); fs.mkdirSync(sdk);
    assert.throws(() => buildJitEmitter(sdk, output, {MOM_AOT_BUILD_ONLY: "1"}), /SYCL headers/); assert.ok(!fs.existsSync(output));
  });
  test("symlink aliases cannot hide output inside the selected SDK", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const root = directory(t), sdk = path.join(root, "sdk"), alias = path.join(root, "alias"); fs.mkdirSync(sdk); fs.symlinkSync(sdk, alias, "dir");
    assert.throws(() => buildJitEmitter(sdk, path.join(alias, "output"), {MOM_AOT_BUILD_ONLY: "1"}), /overlaps/);
  });
  test("CLI requires an explicit emitter stage and all arguments", () => {
    const r = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/native-jit-sdk.js"), "all"], {encoding: "utf8", timeout: 5000});
    assert.equal(r.status, 1); assert.match(r.stderr, /Usage:/); assert.equal(r.stdout, "");
  });
  test("builds a real SDK-independent emitter and rejects malformed requests", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const sdk = process.env["MOM_TEST_DPCPP_SDK"] || "/opt/dpcpp";
    if (!fs.existsSync(path.join(sdk, "include/sycl/sycl.hpp"))) {t.skip("selected SYCL headers unavailable for the CPU-only emitter build"); return;}
    t.mock.method(console, "log", () => undefined);
    const root = directory(t), targetHeader = path.join(root, "target-only.h"); write(targetHeader, "#define TARGET_ONLY 1\n");
    const r = buildJitEmitter(sdk, path.join(root, "emitter"), {...process.env, MOM_AOT_BUILD_ONLY: "1"}, [targetHeader]);
    assert.equal(r.executed, false); assert.equal(r.publicationQualified, false); assert.equal(r.fixtureSha256, fileHash(r.fixture));
    const outputs = new Set();
    for (const algo of ["kawpow", "firopow", "evrprogpow", "meowpow"]) {
      const a = spawnSync(r.fixture, ["--emit-source", algo], {encoding: "utf8", timeout: 10000});
      const b = spawnSync(r.fixture, ["--emit-source", algo], {encoding: "utf8", timeout: 10000});
      assert.equal(a.status, 0, a.stderr); assert.equal(b.stdout, a.stdout); assert.ok(a.stdout.includes("sycl/sycl.hpp")); outputs.add(a.stdout);
    }
    assert.equal(outputs.size, 4);
    for (const args of [[], ["spirv"], ["--emit-source"], ["--emit-source", "not-an-algorithm"]]) {
      const child = spawnSync(r.fixture, args, {encoding: "utf8", timeout: 10000}); assert.equal(child.status, 1); assert.equal(child.signal, null);
    }
    assert.ok(Object.keys(r.inputs).some(file => file.includes("sycl/sycl.hpp")), "actual SDK headers must be pinned");
    assert.equal(r.inputs[targetHeader], fileHash(targetHeader)); assert.deepEqual(r.targetHeaders, [targetHeader]);
  });
});

test.describe("Linux GNU source-JIT resource admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("legacy target omission still selects the original Windows ABI", () => {
    assert.deepEqual(resourceTarget(), resourceTarget("x86_64-w64-windows-gnu"));
    assert.equal(resourceTarget().roots.length, 3); assert.equal(resourceTarget().platform, "windows-gnu");
  });
  test("Linux explicitly selects both common and architecture-specific glibc roots", () => {
    const p = resourceTarget("x86_64-unknown-linux-gnu");
    assert.equal(p.platform, "linux-gnu"); assert.deepEqual(p.roots, ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target", "include/mom-system"]);
  });
  test("only the Windows auxiliary ABI suppresses OpenCL's host-vector aliases", () => {
    assert.deepEqual(resourceTarget("x86_64-unknown-linux-gnu").undefines, []);
    assert.deepEqual(resourceTarget().undefines, ["__SSE__", "__SSE2__", "__MMX__", "__AVX__"]);
  });
  test("unknown targets cannot silently select a host ABI", () => {
    for (const target of ["", "linux", "x86_64-linux-gnu", "aarch64-linux-gnu"]) {assert.throws(() => resourceTarget(target), /Unsupported/);}
  });
  test("target policies cannot mutate one another or later calls", () => {
    const p = resourceTarget(); p.roots.length = 0; p.undefines.push("bad");
    assert.equal(resourceTarget().roots.length, 3); assert.equal(resourceTarget("x86_64-unknown-linux-gnu").roots.length, 4);
    assert.equal(resourceTarget().undefines.length, 4);
  });
  test("validates the full Linux profile with the same receipt and manifest rules", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.fixture) {return;}
    assert.equal(validateProfile(p), p);
  });
  test("a Linux profile cannot omit the common system root", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.headerRoots) {return;}
    const roots = p.headerRoots;
    assert.throws(() => validateProfile({...p, headerRoots: roots.slice(0, 3)}), /four header roots/);
  });
  test("Windows cannot accidentally inherit Linux's extra system root", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.fixture) {return;}
    assert.throws(() => validateProfile({...p, target: "x86_64-w64-windows-gnu"}), /three header roots/);
  });
  test("a common-root alias cannot replace the required virtual namespace", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.headerRoots) {return;}
    const roots = p.headerRoots;
    assert.throws(() => validateProfile({...p, headerRoots: roots.map((r, i) => i === 3 ? {...r, virtual: "include/wrong"} : r)}), /layout/);
  });
  test("the actual Linux probe discovers common headers and records the correct ABI", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.fixture) {return;}
    const r = prepare(p, path.join(path.dirname(p.fixture), "linux-resources"));
    assert.equal(r.platform, "linux-gnu"); assert.equal(r.target, p.target); assert.deepEqual(r.undefines, []);
    assert.ok(r.files.some(row => row.root === "gnu" && row.virtual === "include/mom-system/common.h"));
  });
  test("common Linux headers still require an exact pinned receipt entry", t => {
    const p = actualProfile(t, "x86_64-unknown-linux-gnu"); if (!p.receipt || !p.fixture || !p.gnuRoot) {return;}
    const r = JSON.parse(fs.readFileSync(p.receipt, "utf8")); delete r.inputs[path.join(p.gnuRoot, "include/mom-system/common.h")];
    fs.writeFileSync(p.receipt, JSON.stringify(r));
    const output = path.join(path.dirname(p.fixture), "unadmitted-common");
    assert.throws(() => prepare(p, output), /absent from the pinned SDK receipt/); assert.ok(!fs.existsSync(output));
  });
  test("the existing Windows profile keeps its original advertised ABI", t => {
    const p = actualProfile(t); if (!p.fixture) {return;}
    const r = prepare(p, path.join(path.dirname(p.fixture), "windows-resources"));
    assert.equal(r.platform, "windows-gnu"); assert.equal(r.target, "x86_64-w64-windows-gnu");
  });
});

test.describe("Windows GNU static-JIT resource producer", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("accepts only the fixed three-root profile shape", () => {
    const profile = {version: 1, fixture: "/build/fixture", clang: "/build/clang", receipt: "/build/receipt", gnuRoot: "/build/gnu", baseRoot: "/build/base",
      noticeRoot: "/build/gnu", syclInclude: "/build/base/include/sycl", clangBuiltinInclude: "/build/base/lib/clang", baseManifest: "/build/base-manifest",
      headerRoots: ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target"].map((virtual) => ({path: "/build/gnu/" + virtual, virtual})),
      required: {spirv32: {virtual: "lib/spirv32.bc", sha256: "a".repeat(64)}, spirv64: {virtual: "lib/spirv64.bc", sha256: "b".repeat(64)}, libdevice: {virtual: "lib/libdevice.bc", sha256: "c".repeat(64)}},
      notices: [{source: "usr/share/LICENSE", virtual: "usr/share/LICENSE", sha256: "a".repeat(64)}]};
    assert.equal(validateProfile(profile).version, 1);
    assert.throws(() => validateProfile({...profile, headerRoots: profile.headerRoots.slice(1)}), /three header roots/);
    assert.throws(() => validateProfile({...profile, headerRoots: profile.headerRoots.map((root, index) => index === 0 ? {...root, virtual: "include/other"} : root)}), /layout/);
  });

  test("rejects profile paths and notice names that could escape", () => {
    const profile = {version: 1, fixture: "relative", clang: "/build/clang", receipt: "/build/receipt", gnuRoot: "/build/gnu", baseRoot: "/build/base",
      noticeRoot: "/build/gnu", syclInclude: "/build/base/include/sycl", clangBuiltinInclude: "/build/base/lib/clang", baseManifest: "/build/base-manifest",
      headerRoots: ["a", "b", "c"].map((pathValue, index) => ({path: "/build/gnu/" + pathValue, virtual: ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target"][index]})),
      required: {spirv32: {virtual: "lib/spirv32.bc", sha256: "a".repeat(64)}, spirv64: {virtual: "lib/spirv64.bc", sha256: "b".repeat(64)}, libdevice: {virtual: "lib/libdevice.bc", sha256: "c".repeat(64)}},
      notices: [{source: "../LICENSE", virtual: "LICENSE", sha256: "a".repeat(64)}]};
    assert.throws(() => validateProfile(profile), /absolute fixture/);
    assert.throws(() => validateProfile({...profile, fixture: "/build/fixture", notices: [{source: "../LICENSE", virtual: "LICENSE", sha256: "a".repeat(64)}]}), /relative notice source/);
    assert.throws(() => validateProfile({...profile, fixture: "/build/fixture", notices: [{source: "C:license", virtual: "LICENSE", sha256: "a".repeat(64)}]}), /relative notice source/);
    assert.throws(() => validateProfile({...profile, fixture: "/build/fixture", notices: [{source: "license\n", virtual: "LICENSE", sha256: "a".repeat(64)}]}), /relative notice source/);
  });

  test("keeps the real device-only SYCL probe flags and separate roots in the producer", () => {
    assert.match(producerSource, /"-fsycl", "-fsycl-device-only"/);
    assert.match(producerSource, /"-nostdinc", "-nostdinc\+\+"/);
    assert.match(producerSource, /gnuRoot/); assert.match(producerSource, /baseRoot/);
    assert.match(producerSource, /clangBuiltinInclude/); assert.match(producerSource, /syclInclude/);
  });

  test("rejects missing required device resources before touching output", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.required) {return;}
    const output = path.join(path.dirname(profile.fixture), "missing-required");
    assert.throws(() => prepare({...profile, required: {...profile.required, libdevice: undefined}}, output), /Missing required libdevice/);
    assert.equal(fs.existsSync(output), false);
  });

  test("admits the explicit manifest and emits a deterministic bounded overlay", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.clang || !profile.receipt || !profile.gnuRoot || !profile.baseRoot || !profile.noticeRoot || !profile.syclInclude ||
        !profile.clangBuiltinInclude || !profile.baseManifest || !profile.headerRoots || !profile.required || !profile.notices) {return;}
    const root = path.dirname(profile.fixture), output = path.join(root, "resources");
    const first = prepare(profile, output), second = prepare(profile, path.join(root, "resources-second"));
    assert.deepEqual(first.generated, second.generated);
    assert.deepEqual(first.dependencies, second.dependencies);
    assert.equal(first.publicationQualified, false);
    assert.ok(first.files.some((file) => file.root === "gnu"), "the explicit GNU closure must contribute resources");
    assert.equal(fs.readFileSync(path.join(output, "overlay/include/mom-cxx/mom.h"), "utf8"), "#define MOM_HEADER 1\n");
    assert.equal(fs.readFileSync(path.join(output, "overlay/include/mom-cxx/backward/backward.h"), "utf8"), "#define MOM_BACKWARD 1\n");
    assert.equal(fs.readFileSync(path.join(output, "notices/usr/share/licenses/mom/LICENSE"), "utf8"), "test notice\n");
    assert.equal(fs.existsSync(path.join(output, "work")), false);
    assert.equal(fs.readFileSync(path.join(output, "manifest.json"), "utf8").includes(root), false);
  });

  test("fails closed before creating output when the pinned fixture changes", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture) {return;}
    fs.appendFileSync(profile.fixture, "changed");
    const output = path.join(path.dirname(profile.fixture), "changed-output");
    assert.throws(() => prepare(profile, output), /fixture\/compiler changed/);
    assert.equal(fs.existsSync(output), false);
  });

  test("producer receipts still pin the complete emitter recipe", t => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.receipt) {return;}
    const receipt = JSON.parse(fs.readFileSync(profile.receipt, "utf8"));
    delete receipt.inputs[path.resolve(__dirname, "..", buildOnlyInput)];
    fs.writeFileSync(profile.receipt, JSON.stringify(receipt));
    const output = path.join(path.dirname(profile.fixture), "missing-recipe-input");
    assert.throws(() => prepare(profile, output), /does not pin maintained source input/);
    assert.equal(fs.existsSync(output), false);
  });

  test("preserves compiler diagnostics while removing a failed staging tree", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.gnuRoot || !profile.receipt) {return;}
    const header = path.join(profile.gnuRoot, "include/mom-cxx/mom.h");
    fs.writeFileSync(header, "#error MOM_EXPECTED_JIT_HEADER_DIAGNOSTIC\n");
    const receipt = JSON.parse(fs.readFileSync(profile.receipt, "utf8"));
    receipt.inputs[header] = fileHash(header);
    fs.writeFileSync(profile.receipt, JSON.stringify(receipt));
    const output = path.join(path.dirname(profile.fixture), "compiler-failure");
    assert.throws(() => prepare(profile, output), /dependency probe failed[\s\S]*MOM_EXPECTED_JIT_HEADER_DIAGNOSTIC/);
    assert.equal(fs.existsSync(output), false);
    assert.equal(fs.readdirSync(path.dirname(output)).some(name => name.startsWith(".mom-sycl-jit-resource-")), false);
  });

  test("rejects an output nested under an admitted input root before staging", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.gnuRoot) {return;}
    const output = path.join(profile.gnuRoot, "generated");
    assert.throws(() => prepare(profile, output), /outside every admitted input root/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects overlapping GNU and base roots before staging", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.gnuRoot) {return;}
    const output = path.join(path.dirname(profile.fixture), "overlap");
    assert.throws(() => prepare({...profile, baseRoot: profile.gnuRoot}, output), /non-overlapping/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects a base manifest case-fold collision before staging", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.baseRoot || !profile.baseManifest) {return;}
    const duplicate = path.join(profile.baseRoot, "lib/SPIRV32.bc"); write(duplicate, "different\n");
    const manifest = JSON.parse(fs.readFileSync(profile.baseManifest, "utf8"));
    manifest.files.push(baseEntry(profile.baseRoot, "lib/SPIRV32.bc", "LIB/spirv32.bc")); fs.writeFileSync(profile.baseManifest, JSON.stringify(manifest));
    const output = path.join(path.dirname(profile.fixture), "case-collision");
    assert.throws(() => prepare(profile, output), /duplicate/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects a forbidden binary payload from the explicit base manifest", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.baseRoot || !profile.baseManifest) {return;}
    const forbidden = path.join(profile.baseRoot, "lib/bad.dll"); write(forbidden, "not a runtime\n");
    const manifest = JSON.parse(fs.readFileSync(profile.baseManifest, "utf8"));
    manifest.files.push(baseEntry(profile.baseRoot, "lib/bad.dll", "lib/bad.dll")); fs.writeFileSync(profile.baseManifest, JSON.stringify(manifest));
    const output = path.join(path.dirname(profile.fixture), "forbidden-binary");
    assert.throws(() => prepare(profile, output), /Forbidden/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects a base symlink escaping its admitted root", (t) => {
    if (process.platform === "win32") {t.skip("symlink fixture requires Windows developer-link privileges"); return;}
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.baseRoot || !profile.baseManifest) {return;}
    const outside = path.join(path.dirname(profile.baseRoot), "outside.h"); write(outside, "outside\n");
    const escaped = path.join(profile.baseRoot, "lib/escaped.h"); fs.symlinkSync(outside, escaped);
    const manifest = JSON.parse(fs.readFileSync(profile.baseManifest, "utf8"));
    manifest.files.push({source: "lib/escaped.h", virtual: "include/escaped.h", sha256: fileHash(outside)}); fs.writeFileSync(profile.baseManifest, JSON.stringify(manifest));
    const output = path.join(path.dirname(profile.fixture), "escaped-link");
    assert.throws(() => prepare(profile, output), /escapes/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects a GNU dependency absent from the pinned receipt", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.receipt) {return;}
    const receipt = JSON.parse(fs.readFileSync(profile.receipt, "utf8"));
    delete receipt.inputs[path.resolve(path.dirname(profile.gnuRoot || ""), "gnu-package/include/mom-cxx/mom.h")];
    const mom = Object.keys(receipt.inputs).find(file => file.endsWith("/include/mom-cxx/mom.h"));
    if (mom) {delete receipt.inputs[mom];}
    fs.writeFileSync(profile.receipt, JSON.stringify(receipt));
    const output = path.join(path.dirname(profile.fixture), "missing-g-n-u-input");
    assert.throws(() => prepare(profile, output), /absent from the pinned SDK receipt/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects notices that exceed the combined resource bound", (t) => {
    const profile = actualProfile(t);
    if (!profile.fixture || !profile.noticeRoot) {return;}
    const huge = path.join(profile.noticeRoot, "HUGE"); fs.writeFileSync(huge, Buffer.alloc((64 << 20) + 1));
    const output = path.join(path.dirname(profile.fixture), "notice-bound");
    const notices = [...(profile.notices || []), {source: "HUGE", virtual: "usr/share/licenses/mom/HUGE", sha256: fileHash(huge)}];
    assert.throws(() => prepare({...profile, notices}, output), /exceed bounds/);
    assert.equal(fs.existsSync(output), false);
  });
});
