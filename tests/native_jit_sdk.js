"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {jitSdkConfig, jitDependencyFiles, jitConfiguredTools, jitConfiguredCompilerFiles, jitConfiguredCompilers, jitCompilerIdentities,
  jitSupportLibraries, buildJitSdk} = require("../scripts/build-native-jit-sdk");
const {jitSdkPlan, jitEmitterPlan} = require("../scripts/native-jit-sdk");
const {compilerArchives} = require("../scripts/native-compiler-isolation");
const {fileHash, buildFileInputs} = require("../scripts/build-source-inputs");
const {windowsIntelRuleOverride, windowsJitToolchainPlan, prepareWindowsJitToolchain} = require("../scripts/prepare-native-jit-toolchain");
const {sdkComponents} = require("../scripts/prepare-native-sdk-sources");

function config() {return {version: 1, target: "linux", sources: "/sources.json", sdk: "/sdk", resources: "/resources", tools: {cmake: "/cmake", ninja: "/ninja"}, jobs: 2};}
/** @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-sdk-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const file = path.join(root, "deps.log");
  return {root, file, write: (/** @type {string} */ text) => fs.writeFileSync(file, text)};
}
/** @param {string[]} files @param {string} [name] */
function record(files, name = "object.o") {return name + ": #deps " + files.length + ", deps mtime 123456789 (VALID)\n" + files.map(file => "    " + file + "\n").join("");}

function windowsDescriptor() {
  return {version: 1, tools: {clangCl: "/tools/clang-cl", lldLink: "/tools/lld-link", shermaot: "/tools/shermes", llvmNm: "/tools/llvm-nm", llvmReadobj: "/tools/llvm-readobj", clangResourceInclude: "/tools/include"},
    hermes: {source: "/hermes", build: "/hermes-build", shermaot: "/tools/shermes"},
    windowsSdk: {overlay: "/vfs.json", sourceRoot: "/os", includeRoots: ["/virtual/msvc", "/virtual/ucrt"], libraryRoots: ["/os/lib"],
      msvcLibraries: ["/os/lib/libcmt.lib"], ucrtLibraries: ["/os/lib/libucrt.lib"], importLibraries: ["/os/lib/kernel32.lib"], allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: "/include", ssl: "/ssl.lib", crypto: "/crypto.lib", libuv: "/uv.lib", zlib: "/z.lib"}};
}

function windowsLlvmDescriptor() {
  const full = windowsDescriptor();
  return {version: full.version,
    tools: {clangCl: full.tools.clangCl, lldLink: full.tools.lldLink, llvmNm: full.tools.llvmNm,
      clangResourceInclude: full.tools.clangResourceInclude},
    windowsSdk: {overlay: full.windowsSdk.overlay, sourceRoot: full.windowsSdk.sourceRoot,
      includeRoots: full.windowsSdk.includeRoots, libraryRoots: full.windowsSdk.libraryRoots,
      msvcLibraries: full.windowsSdk.msvcLibraries, ucrtLibraries: full.windowsSdk.ucrtLibraries,
      importLibraries: full.windowsSdk.importLibraries}};
}

/** Filesystem-only SDK fixture: none of these placeholder tools is executed.
 * @param {import('node:test').TestContext} t */
function crossFixture(t) {
  const f = fixture(t), d = windowsDescriptor();
  const put = (/** @type {string} */ name, data = "fixture") => {const file = path.join(f.root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return file;};
  for (const name of ["sdk/bin/clang-cl", "sdk/bin/llvm-ar", "sdk/bin/llvm-spirv", "native/llvm-tblgen", "native/clang-tblgen", "tools/lld-link", "tools/llvm-rc", "tools/llvm-mt", "tools/llvm-ml"]) {put(name);}
  const header = put("os/include/a.h"), library = put("os/lib/Kernel32.Lib"); d.tools.lldLink = path.join(f.root, "tools/lld-link");
  d.windowsSdk.sourceRoot = path.join(f.root, "os"); d.windowsSdk.libraryRoots = [path.join(f.root, "os/lib")];
  d.windowsSdk.overlay = put("vfs.json", JSON.stringify({version: 0, "case-sensitive": false, roots: [
    {type: "file", name: "/virtual/msvc/a.h", "external-contents": header}, {type: "file", name: "/unused/absent.h", "external-contents": "/nonexistent.h"}]}));
  const descriptor = put("descriptor.json", JSON.stringify(d));
  return {...f, header, library, descriptor, sdk: path.join(f.root, "sdk"), native: path.join(f.root, "native"), output: path.join(f.root, "new")};
}

test.describe("Original compiler SDK configuration", () => {
  test("Linux and Windows select the same bounded producer contract", () => {
    assert.deepEqual(jitSdkConfig(config()), config());
    const c = {...config(), target: "windows", tools: {...config().tools, toolchain: "/toolchain.cmake"}};
    assert.deepEqual(jitSdkConfig(c), c);
  });
  test("explicit host C and C++ overrides are paired and remain bounded literals", () => {
    const linux = {...config(), tools: {...config().tools, cc: "/intel/icx", cxx: "/intel/icpx"}};
    assert.deepEqual(jitSdkConfig(linux), linux);
    const windows = {...config(), target: "windows", tools: {...linux.tools, toolchain: "/toolchain.cmake"}};
    assert.deepEqual(jitSdkConfig(windows), windows);
    for (const tools of [{...config().tools, cc: "/intel/icx"}, {...config().tools, cxx: "/intel/icpx"},
      {...config().tools, cc: "icx", cxx: "/intel/icpx"}, {...config().tools, cc: "/intel/icx;unsafe", cxx: "/intel/icpx"}]) {
      assert.throws(() => jitSdkConfig({...config(), tools}), /Incomplete/);
    }
  });
  test("missing fields cannot silently select host defaults", () => {
    for (const name of Object.keys(config())) {const c = /** @type {Record<string, unknown>} */ (config()); delete c[name]; assert.throws(() => jitSdkConfig(c));}
  });
  test("unknown flags and setup hooks are not accepted", () => {
    assert.throws(() => jitSdkConfig({...config(), flags: "-unsafe"}));
    assert.throws(() => jitSdkConfig({...config(), tools: {...config().tools, shell: "/shell"}}));
  });
  test("unrecognized targets and versions are rejected", () => {
    for (const target of ["darwin", "", null]) {assert.throws(() => jitSdkConfig({...config(), target}));}
    assert.throws(() => jitSdkConfig({...config(), version: 2}));
  });
  test("parallelism cannot exceed the explicit supported bound", () => {
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => jitSdkConfig({...config(), jobs}));}
  });
  test("Windows needs an explicit cross toolchain", () => {
    assert.throws(() => jitSdkConfig({...config(), target: "windows"}), /Incomplete/);
  });
  test("relative and control-containing selected paths are rejected", () => {
    for (const sdk of ["sdk", "/sdk\nother", "/sdk\0other"]) {assert.throws(() => jitSdkConfig({...config(), sdk}));}
  });
  test("additional toolchain input lists stay bounded and absolute", () => {
    assert.equal(jitSdkConfig({...config(), inputs: ["/header"]}).inputs?.[0], "/header");
    for (const inputs of ["/header", ["relative"], Array(16385).fill("/header")]) {assert.throws(() => jitSdkConfig({...config(), inputs}));}
  });
  test("Windows resource.cpp stays on SDK Clang23 independently of host CMake overrides", () => {
    const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/sources/" + name]));
    const sdk = "/sdk", build = "/output/build", sdkCompiler = path.join(sdk, "bin/clang-cl");
    const windows = jitSdkPlan("windows", roots, sdk, "/resources", "/output",
      {cmake: "/tools/cmake", ninja: "/tools/ninja", toolchain: "/tools/windows.cmake"}, 2);
    assert.equal(windows.definitions["CMAKE_CXX_COMPILER"], sdkCompiler);
    assert.equal(windows.definitions["SYCL_JIT_RESOURCE_CLANG"], sdkCompiler);
    const overridden = jitSdkPlan("windows", roots, sdk, "/resources", "/output",
      {cmake: "/tools/cmake", ninja: "/tools/ninja", toolchain: "/tools/windows.cmake", cc: "/intel/icx", cxx: "/intel/icpx"}, 2);
    assert.equal(overridden.definitions["CMAKE_C_COMPILER"], "/intel/icx");
    assert.equal(overridden.definitions["CMAKE_CXX_COMPILER"], "/intel/icpx");
    assert.equal(overridden.definitions["SYCL_JIT_RESOURCE_CLANG"], sdkCompiler);
    assert.deepEqual(overridden.compilerIdentity, {host: {mode: "explicit-host-override", cc: "/intel/icx", cxx: "/intel/icpx"}, embeddedJit23: sdkCompiler});
    assert.ok(overridden.configure.includes("-DSYCL_JIT_RESOURCE_CLANG=" + sdkCompiler));
    assert.notEqual(overridden.definitions["SYCL_JIT_RESOURCE_CLANG"], path.join(build, "bin/clang-cl.exe"));

    const linux = jitSdkPlan("linux", roots, sdk, "/resources", "/output",
      {cmake: "/tools/cmake", ninja: "/tools/ninja"}, 2);
    assert.equal(Object.hasOwn(linux.definitions, "SYCL_JIT_RESOURCE_CLANG"), false);
    assert.equal(jitEmitterPlan(sdk, "/emitter").cxx, path.join(sdk, "bin/clang++"), "the CPU source emitter remains on the selected SDK");
  });
  test("fresh Windows builds keep their own Ninja tree and use the prepared toolchain hook", () => {
    const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/sources/" + name]));
    const plan = jitSdkPlan("windows", roots, "/sdk", "/resources", "/fresh-sdk",
      {cmake: "/tools/cmake", ninja: "/tools/ninja", toolchain: "/prepared/toolchain.cmake"}, 2);
    assert.equal(plan.build, "/fresh-sdk/build");
    assert.equal(plan.definitions["CMAKE_TOOLCHAIN_FILE"], "/prepared/toolchain.cmake");
    assert.equal(Object.hasOwn(plan.definitions, "CMAKE_PROJECT_TOP_LEVEL_INCLUDES"), false,
      "the toolchain owns the hook; no second migration-only override");
    assert.ok(plan.configure.includes(plan.build));
    assert.ok(plan.compile.includes(plan.build));
  });
  test("compiler identities distinguish Intel host version from embedded Clang23", () => {
    const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/sources/" + name]));
    const p = jitSdkPlan("linux", roots, "/sdk", "/resources", "/output",
      {cmake: "/tools/cmake", ninja: "/tools/ninja", cc: "/intel/icx", cxx: "/intel/icpx"}, 2);
    const inputs = {[p.compilerIdentity.host.cc]: "a".repeat(64), [p.compilerIdentity.host.cxx]: "b".repeat(64), [p.compilerIdentity.embeddedJit23]: "c".repeat(64)};
    const versions = {[p.compilerIdentity.host.cc]: "Intel(R) oneAPI DPC++/C++ Compiler 2026.2\n",
      [p.compilerIdentity.host.cxx]: "Intel(R) oneAPI DPC++/C++ Compiler 2026.2\n",
      [p.compilerIdentity.embeddedJit23]: "clang version 23.0.0\n"};
    const identity = jitCompilerIdentities(p, inputs, versions);
    assert.equal(identity.host.mode, "explicit-host-override");
    assert.match(identity.host.cxx.version, /Intel.*2026\.2/);
    assert.equal(identity.embeddedJit23.path, "/sdk/bin/clang++");
    assert.match(identity.embeddedJit23.version, /clang version 23\./);
    const missingEmbedded = {...inputs}; delete missingEmbedded[p.compilerIdentity.embeddedJit23];
    assert.throws(() => jitCompilerIdentities(p, missingEmbedded, versions), /incomplete/);

    const baseline = jitSdkPlan("linux", roots, "/sdk", "/resources", "/output", {cmake: "/tools/cmake", ninja: "/tools/ninja"}, 2);
    const baselineInputs = {[baseline.compilerIdentity.host.cc]: "a".repeat(64), [baseline.compilerIdentity.host.cxx]: "b".repeat(64)};
    assert.throws(() => jitCompilerIdentities(baseline, baselineInputs, {
      [baseline.compilerIdentity.host.cc]: "Intel(R) oneAPI DPC++/C++ Compiler 2026.2\n",
      [baseline.compilerIdentity.host.cxx]: "clang version 23.0.0\n"}), /baseline/);
    assert.throws(() => jitCompilerIdentities(p, inputs, {...versions, [p.compilerIdentity.embeddedJit23]: "Intel 2026.2\n"}), /embedded Clang23/);
  });
  test("CMake cache and generated compiler files must retain the requested paths before building", t => {
    const expected = {CMAKE_C_COMPILER: "/intel/icx", CMAKE_CXX_COMPILER: "/intel/icpx"};
    const cache = "CMAKE_C_COMPILER:FILEPATH=/intel/icx\nCMAKE_CXX_COMPILER:FILEPATH=/intel/icpx\n";
    const generated = {CMAKE_C_COMPILER: 'set(CMAKE_C_COMPILER "/intel/icx")\n', CMAKE_CXX_COMPILER: 'set(CMAKE_CXX_COMPILER "/intel/icpx")\n'};
    assert.deepEqual(jitConfiguredCompilers(cache, expected, generated), expected);
    // A Windows toolchain can shadow a matching cache entry with a normal CMake variable.
    assert.throws(() => jitConfiguredCompilers(cache, expected, {
      ...generated, CMAKE_C_COMPILER: 'set(CMAKE_C_COMPILER "/sdk/bin/clang-cl")\n',
    }), /generated a compiler different/);
    // Cache rewrites are independently rejected too.
    assert.throws(() => jitConfiguredCompilers("CMAKE_C_COMPILER:FILEPATH=/sdk/bin/clang-cl\nCMAKE_CXX_COMPILER:FILEPATH=/sdk/bin/clang-cl\n", expected, generated), /different from the requested/);
    for (const text of ["", "CMAKE_C_COMPILER:FILEPATH=/intel/icx\n", cache + "CMAKE_C_COMPILER:FILEPATH=/other/icx\n",
      "CMAKE_C_COMPILER:FILEPATH=relative\nCMAKE_CXX_COMPILER:FILEPATH=/intel/icpx\n"]) {
      assert.throws(() => jitConfiguredCompilers(text, expected, generated));
    }
    for (const compilerText of ["", 'set(CMAKE_C_COMPILER "relative")\n', generated.CMAKE_C_COMPILER + generated.CMAKE_C_COMPILER]) {
      assert.throws(() => jitConfiguredCompilers(cache, expected, {...generated, CMAKE_C_COMPILER: compilerText}));
    }

    const build = path.join(fixture(t).root, "build"), versionDirectory = path.join(build, "CMakeFiles", "4.2.1");
    fs.mkdirSync(versionDirectory, {recursive: true});
    for (const file of ["CMakeCCompiler.cmake", "CMakeCXXCompiler.cmake"]) {fs.writeFileSync(path.join(versionDirectory, file), "compiler metadata");}
    assert.deepEqual(jitConfiguredCompilerFiles(build), {
      CMAKE_C_COMPILER: path.join(versionDirectory, "CMakeCCompiler.cmake"),
      CMAKE_CXX_COMPILER: path.join(versionDirectory, "CMakeCXXCompiler.cmake"),
    });
    fs.mkdirSync(path.join(build, "CMakeFiles", "4.2.2"));
    assert.throws(() => jitConfiguredCompilerFiles(build), /exactly one/);
  });
  test("no build starts without explicit build-only admission", async () => {
    await assert.rejects(buildJitSdk(config(), "/missing", {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("discovered tools are literal files and generated TableGen stays owned", t => {
    const f = fixture(t); f.write("tool");
    const cache = ["CMAKE_AR:FILEPATH=" + f.file, "MOM_NODE:FILEPATH=" + f.file, "CLANG_TABLEGEN:STRING=clang-tblgen",
      "LLVM_TABLEGEN:FILEPATH=" + path.join(f.root, "build/bin/llvm-tblgen"), "CMAKE_RANLIB:FILEPATH=RANLIB-NOTFOUND", "OTHER:STRING=ignored"].join("\n");
    assert.deepEqual(jitConfiguredTools(cache, path.join(f.root, "build")), [fs.realpathSync(f.file)]);
  });
  test("discovered Python is frozen along with compiler utilities", t => {
    const f = fixture(t); f.write("python");
    assert.deepEqual(jitConfiguredTools("_Python3_EXECUTABLE:INTERNAL=" + f.file, path.join(f.root, "build")), [fs.realpathSync(f.file)]);
  });
  test("every configured archive and link utility is hash-bound against replacement", t => {
    const f = fixture(t), roles = ["CMAKE_AR", "CMAKE_RANLIB", "CMAKE_LINKER", "CMAKE_MAKE_PROGRAM",
        "CMAKE_C_COMPILER_AR", "CMAKE_CXX_COMPILER_AR", "CMAKE_C_COMPILER_RANLIB", "CMAKE_CXX_COMPILER_RANLIB"],
      files = roles.map(role => {
        const file = path.join(f.root, role); fs.writeFileSync(file, role + " original"); return file;
      }), cache = roles.map((role, index) => role + ":FILEPATH=" + files[index]).join("\n"),
      admitted = jitConfiguredTools(cache, path.join(f.root, "build"));
    assert.deepEqual(admitted, files.map(file => fs.realpathSync(file)).sort());
    const before = buildFileInputs(admitted);
    for (const file of admitted) {
      const original = fs.readFileSync(file);
      fs.appendFileSync(file, " replacement");
      const after = buildFileInputs(admitted);
      assert.notEqual(after[file], before[file], "tool replacement escaped the input digest: " + path.basename(file));
      for (const other of admitted) {if (other !== file) {assert.equal(after[other], before[other]);}}
      fs.writeFileSync(file, original);
    }
  });
  test("a missing or relative configured tool fails closed", () => {
    assert.throws(() => jitConfiguredTools("", "/build"), /Missing/);
    assert.throws(() => jitConfiguredTools("MOM_NODE:FILEPATH=node", "/build"), /non-absolute/);
  });
});

test.describe("Hashed Windows compiler archive support paths", () => {
  const archives = ["lib/sycl-jit.lib", ...Array.from({length: 50}, (_, index) => `lib/LLVM${index}.lib`)];
  const msvcSupport = "/admitted SDK/msvc/lib/libcmt.lib", intelSupport = "/admitted runtime/lib/libircmt.lib", support = [msvcSupport, intelSupport];
  const link = ": && lld-link /out:bin/native-dpcpp-jit.exe /implib:lib/native-dpcpp-jit.lib " + archives.join(" ") +
    " \"/admitted SDK/msvc/lib/libcmt.lib\" \"/admitted runtime/lib/libircmt.lib\" && :";
  test("excludes only exact quoted support paths while preserving ordered compiler archives", () => {
    assert.deepEqual(compilerArchives(link, "/original", true, support), archives.map(file => path.join("/original", file)));
  });
  test("derives support paths only from hashed explicit config inputs", () => {
    const hashes = /** @type {Record<string,string>} */ (Object.fromEntries(support.map((file, index) => [file, String(index + 1).repeat(64)])));
    const selected = jitSdkConfig({...config(), target: "windows", tools: {...config().tools, toolchain: "/toolchain"}, inputs: support});
    assert.deepEqual(jitSupportLibraries(selected, hashes), support);
    assert.deepEqual(jitSupportLibraries(jitSdkConfig({...config(), inputs: support}), {}), []);
    assert.throws(() => jitSupportLibraries(jitSdkConfig({...selected, inputs: [msvcSupport, msvcSupport]}), hashes), /duplicates/);
    assert.throws(() => jitSupportLibraries(selected, {[msvcSupport]: "1".repeat(64)}), /not bound/);
  });
  test("rejects unknown, unquoted, partial, or ambiguously quoted support paths", () => {
    for (const changed of [
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/unhashed/libcmt.lib"'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', msvcSupport),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/admitted SDK/msvc/lib/libcmt.lib"suffix'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/admitted SDK"/msvc/lib/libcmt.lib'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/admitted SDK/msvc/lib/libcmt.lib'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/admitted SDK/msvc/lib/libcmt.lib$(bad)"'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', '"/admitted SDK/msvc/lib/libcmt.lib;bad"'),
      link.replace('"/admitted SDK/msvc/lib/libcmt.lib"', "other/libcmt.lib"),
    ]) {assert.throws(() => compilerArchives(changed, "/original", true, support), /Unrecognized/);}
    assert.throws(() => compilerArchives(link, "/original", true, [...support, msvcSupport]), /Unrecognized/);
    assert.throws(() => compilerArchives(link, "/original", false, support), /Unrecognized/);
  });
  test("Linux archive discovery remains strict and unchanged", () => {
    const linuxArchives = ["lib/libsycl-jit.a", ...Array.from({length: 50}, (_, index) => `lib/libLLVM${index}.a`)];
    const linux = "clang++ fixture.o -o bin/native-dpcpp-jit " + linuxArchives.join(" ");
    assert.deepEqual(compilerArchives(linux, "/original"), linuxArchives.map(file => path.join("/original", file)));
    assert.throws(() => compilerArchives(linux.replace("lib/libLLVM0.a", "/external/libLLVM0.a"), "/original"), /Unrecognized/);
  });
});

test.describe("Streaming compiler dependency admission", () => {
  test("unions literal relative, absolute and spaced filenames", async t => {
    const f = fixture(t), external = path.resolve(f.root, "../sdk/a header.h");
    f.write(record(["source.cpp", external]) + "\n" + record(["source.cpp", "other.h"], "other.o"));
    assert.deepEqual(await jitDependencyFiles(f.file, f.root), [external, path.join(f.root, "other.h"), path.join(f.root, "source.cpp")].sort());
  });
  test("CRLF and a final line without newline remain valid", async t => {
    const f = fixture(t); f.write(record(["a.h"]).trimEnd().replaceAll("\n", "\r\n"));
    assert.deepEqual(await jitDependencyFiles(f.file, f.root), [path.join(f.root, "a.h")]);
  });
  test("MSVC PCH consumers and header-free assembly may report zero direct dependencies", async t => {
    const f = fixture(t); f.write(record(["sdk/header.h"], "cmake_pch.obj") + "\n" + record([], "pch-consumer.obj") + "\n" + record([], "assembly.obj"));
    assert.deepEqual(await jitDependencyFiles(f.file, f.root), [path.join(f.root, "sdk/header.h")]);
  });
  test("zero-dependency records alone are never a complete compiler closure", async t => {
    const f = fixture(t); f.write(record([], "assembly.obj")); await assert.rejects(jitDependencyFiles(f.file, f.root), /incomplete/);
  });
  test("records can span stream chunks without losing files", async t => {
    const f = fixture(t), files = Array.from({length: 8000}, (_, i) => "include/long-header-" + i + ".h"); f.write(record(files));
    assert.equal((await jitDependencyFiles(f.file, f.root)).length, files.length);
  });
  test("an early EOF cannot certify a truncated dependency record", async t => {
    const f = fixture(t); f.write(record(["a.h", "b.h"]).replace("    b.h\n", ""));
    await assert.rejects(jitDependencyFiles(f.file, f.root), /Incomplete/);
  });
  test("record boundaries cannot hide missing dependencies", async t => {
    const f = fixture(t); f.write(record(["a.h", "b.h"]).replace("    b.h\n", "\n") + record(["c.h"], "other.o"));
    await assert.rejects(jitDependencyFiles(f.file, f.root), /Incomplete/);
  });
  test("unexpected extra dependencies fail closed", async t => {
    const f = fixture(t); f.write(record(["a.h"]) + "    b.h\n");
    await assert.rejects(jitDependencyFiles(f.file, f.root), /Unrecognized/);
  });
  test("stale Ninja records are not success evidence", async t => {
    const f = fixture(t); f.write(record(["a.h"]).replace("VALID", "STALE"));
    await assert.rejects(jitDependencyFiles(f.file, f.root), /stale/);
  });
  test("an empty inventory is not a compiler closure", async t => {
    const f = fixture(t); f.write("\n"); await assert.rejects(jitDependencyFiles(f.file, f.root), /incomplete/);
  });
  test("per-record counts remain bounded", async t => {
    const f = fixture(t); f.write(record(["a.h"]).replace("#deps 1", "#deps 65537"));
    await assert.rejects(jitDependencyFiles(f.file, f.root), /count/);
  });
  test("oversized lines cannot grow the carry buffer indefinitely", async t => {
    const f = fixture(t); f.write(record(["a".repeat(17000)])); await assert.rejects(jitDependencyFiles(f.file, f.root), /line/);
  });
  test("control characters and malformed records are rejected", async t => {
    const f = fixture(t);
    for (const data of [record(["a\0b"]), "unexpected\n"]) {f.write(data); await assert.rejects(jitDependencyFiles(f.file, f.root));}
  });
  test("symlink logs cannot redirect admission", async t => {
    const f = fixture(t); f.write(record(["a.h"])); const alias = path.join(f.root, "alias");
    try {fs.symlinkSync(f.file, alias);} catch (error) {
      if (process.platform === "win32" && error instanceof Error && "code" in error && ["EPERM", "EACCES"].includes(String(error.code))) {t.skip("Windows symlink privilege unavailable"); return;}
      throw error;
    }
    await assert.rejects(jitDependencyFiles(alias, f.root), /regular/);
  });
  test("oversized sparse logs are rejected before reading", async t => {
    const f = fixture(t); f.write("x"); fs.truncateSync(f.file, (256 << 20) + 1);
    await assert.rejects(jitDependencyFiles(f.file, f.root), /bounds/);
  });
});

test.describe("Shared Windows source-JIT cross toolchain", {skip: process.platform !== "linux" && "Linux cross-build preparation"}, () => {
  test("accepts the exact partial descriptor emitted by the Windows LLVM composer", () => {
    const descriptor = windowsLlvmDescriptor(), plan = windowsJitToolchainPlan(descriptor, "/sdk", "/native", "/output");
    assert.deepEqual(Object.keys(plan.config).sort(), ["tools", "version", "windowsSdk"]);
    assert.deepEqual(Object.keys(plan.config.tools).sort(), ["clangCl", "clangResourceInclude", "lldLink", "llvmNm"]);
    assert.equal(plan.config.windowsSdk.overlay, descriptor.windowsSdk.overlay);
    assert.equal(plan.compilerTools.CMAKE_LINKER, descriptor.tools.lldLink);
    for (const changed of [
      {...descriptor, tools: {...descriptor.tools, clangCl: undefined}},
      {...descriptor, tools: {...descriptor.tools, lldLink: "relative/lld-link"}},
      {...descriptor, windowsSdk: {...descriptor.windowsSdk, libraryRoots: []}},
      {...descriptor, windowsSdk: {...descriptor.windowsSdk, msvcLibraries: undefined}},
    ]) {assert.throws(() => windowsJitToolchainPlan(changed, "/sdk", "/native", "/output"));}
  });
  test("continues to accept the full controller descriptor through the shared LLVM validator", () => {
    const full = windowsDescriptor(), partialPlan = windowsJitToolchainPlan(windowsLlvmDescriptor(), "/sdk", "/native", "/output"),
      fullPlan = windowsJitToolchainPlan(full, "/sdk", "/native", "/output");
    assert.deepEqual(partialPlan.config.tools, {clangCl: full.tools.clangCl, lldLink: full.tools.lldLink,
      llvmNm: full.tools.llvmNm, clangResourceInclude: full.tools.clangResourceInclude});
    assert.deepEqual(fullPlan.config, full);
  });
  test("an explicit Intel host pair preserves the independent original SDK generators", () => {
    const f = windowsDescriptor(), host = {cc: "/intel/bin/icx-cl", cxx: "/intel/bin/icx-cl"};
    const p = windowsJitToolchainPlan(f, "/sdk", "/native", "/output", host);
    assert.equal(p.compilerTools.CMAKE_C_COMPILER, host.cc);
    assert.equal(p.compilerTools.CMAKE_CXX_COMPILER, host.cxx);
    assert.equal(p.compilerTools.LLVM_SPIRV, "/sdk/bin/llvm-spirv");
    assert.equal(p.compilerTools.LLVM_TABLEGEN, "/native/llvm-tblgen");
    assert.ok(p.toolchain.includes(f.tools.clangResourceInclude));
    for (const pair of [{cc: host.cc}, {cc: "relative", cxx: host.cxx}, {...host, cc: "/intel;other"}]) {
      assert.throws(() => windowsJitToolchainPlan(f, "/sdk", "/native", "/output", pair), /host compiler/);
    }
  });
  test("a generic explicit Clang host keeps the manifest tool and original link rules", () => {
    const host = {cc: "/clang/bin/clang-cl", cxx: "/clang/bin/clang-cl"},
      plan = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/output", host);
    assert.equal(plan.compilerTools.CMAKE_C_COMPILER, host.cc);
    assert.equal(plan.compilerTools.CMAKE_MT, "/tools/llvm-mt");
    assert.equal(plan.ruleOverride, undefined);
    assert.equal(plan.ruleOverrideText, undefined);
  });
  test("Intel direct-LLD host mode does not require the unused manifest tool", t => {
    const f = crossFixture(t), hostCompiler = path.join(f.root, "intel/icx-cl"), manifestTool = path.join(f.root, "tools/llvm-mt");
    fs.mkdirSync(path.dirname(hostCompiler), {recursive: true}); fs.writeFileSync(hostCompiler, "fixture compiler"); fs.unlinkSync(manifestTool);
    const host = {cc: hostCompiler, cxx: hostCompiler}, plan = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/output", host),
      prepared = prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, f.output, {MOM_AOT_BUILD_ONLY: "1"}, host);
    assert.equal(Object.hasOwn(plan.compilerTools, "CMAKE_MT"), false);
    assert.equal(Object.hasOwn(prepared.inputs, manifestTool), false);
    assert.match(plan.ruleOverrideText || "", /set\(CMAKE_CXX_LINK_EXECUTABLE \[==\[<CMAKE_LINKER>/);
    assert.doesNotMatch(plan.ruleOverrideText || "", /CMAKE_MT|vs_link_exe/);

    const source = path.join(f.root, "cmake-project"), analysis = path.join(source, "lib/Analysis"), build = path.join(f.root, "cmake-build");
    fs.mkdirSync(analysis, {recursive: true});
    fs.writeFileSync(path.join(source, "CMakeLists.txt"), "cmake_minimum_required(VERSION 3.24)\nproject(NoManifest LANGUAGES CXX)\nadd_subdirectory(lib/Analysis)\nadd_executable(no_manifest main.cpp)\n");
    fs.writeFileSync(path.join(analysis, "CMakeLists.txt"), "add_library(heat_utils OBJECT HeatUtils.cpp)\n");
    fs.writeFileSync(path.join(analysis, "HeatUtils.cpp"), "int heat_utils() { return 0; }\n");
    fs.writeFileSync(path.join(source, "main.cpp"), "int main() { return 0; }\n");
    const configured = spawnSync("cmake", ["-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_TOOLCHAIN_FILE=" + prepared.toolchain,
      "-DCMAKE_CXX_COMPILER_FORCED=TRUE", "-DCMAKE_CXX_COMPILER_WORKS=TRUE", "-DCMAKE_CXX_COMPILER_ID=IntelLLVM",
      "-DCMAKE_CXX_COMPILER_VERSION=2026.2.0", "-DCMAKE_CXX_COMPILER_FRONTEND_VARIANT=MSVC"],
    {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    assert.equal(configured.status, 0, `${configured.stdout}\n${configured.stderr}`);
    const generatedRules = fs.readFileSync(path.join(build, "CMakeFiles/rules.ninja"), "utf8");
    assert.match(generatedRules, /lld-link/);
    assert.doesNotMatch(generatedRules, /llvm-mt|vs_link_exe/);
    assert.match(generatedRules, /\/Tp\$in/, "Intel must receive the source as an explicit filename, not a /w option");
  });
  test("shared Intel linker rules preserve source operands unless explicitly opted in", t => {
    const f = fixture(t), rules = path.join(f.root, "rules.cmake"), script = path.join(f.root, "check.cmake"), result = path.join(f.root, "rules.txt");
    fs.writeFileSync(rules, windowsIntelRuleOverride("", []));
    fs.writeFileSync(script, [
      'set(CMAKE_C_COMPILER_ID "IntelLLVM")', 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")',
      'set(CMAKE_C_COMPILE_OBJECT "cc <FLAGS> -c <SOURCE>")', 'set(CMAKE_CXX_COMPILE_OBJECT "cxx <FLAGS> -c <SOURCE>")',
      `include("${rules}")`,
      'if(NOT CMAKE_C_COMPILE_OBJECT STREQUAL "cc <FLAGS> -c <SOURCE>")', 'message(FATAL_ERROR "C source rule changed by default")', "endif()",
      'if(NOT CMAKE_CXX_COMPILE_OBJECT STREQUAL "cxx <FLAGS> -c <SOURCE>")', 'message(FATAL_ERROR "C++ source rule changed by default")', "endif()",
      `file(WRITE "${result}" "\${CMAKE_C_COMPILE_OBJECT}\\n\${CMAKE_CXX_COMPILE_OBJECT}\\n")`,
    ].join("\n"));
    const run = spawnSync("cmake", ["-P", script], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
    assert.equal(run.status, 0, run.stdout + run.stderr);
    assert.equal(fs.readFileSync(result, "utf8"), "cc <FLAGS> -c <SOURCE>\ncxx <FLAGS> -c <SOURCE>\n");
    assert.throws(() => windowsIntelRuleOverride("", [], /** @type {boolean} */ (/** @type {unknown} */ ("true"))), /source operands/);
  });
  test("opted-in Intel source operands cover both languages and repeated hooks are idempotent", t => {
    const f = fixture(t), rules = path.join(f.root, "rules.cmake"), script = path.join(f.root, "check.cmake"), result = path.join(f.root, "rules.txt");
    fs.writeFileSync(rules, windowsIntelRuleOverride("", [], true));
    fs.writeFileSync(script, [
      'set(CMAKE_C_COMPILER_ID "IntelLLVM")', 'set(CMAKE_CXX_COMPILER_ID "Clang")',
      'set(CMAKE_C_COMPILE_OBJECT "cc <FLAGS> -c <SOURCE>")', 'set(CMAKE_CXX_COMPILE_OBJECT "cxx <FLAGS> -c <SOURCE>")',
      `include("${rules}")`,
      'if(NOT CMAKE_CXX_COMPILE_OBJECT STREQUAL "cxx <FLAGS> -c <SOURCE>")', 'message(FATAL_ERROR "Clang rule changed")', "endif()",
      'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")', `include("${rules}")`, `include("${rules}")`,
      `file(WRITE "${result}" "\${CMAKE_C_COMPILE_OBJECT}\\n\${CMAKE_CXX_COMPILE_OBJECT}\\n")`,
    ].join("\n"));
    const run = spawnSync("cmake", ["-P", script], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
    assert.equal(run.status, 0, run.stdout + run.stderr);
    assert.equal(fs.readFileSync(result, "utf8"), "cc <FLAGS> -c /Tc<SOURCE>\ncxx <FLAGS> -c /Tp<SOURCE>\n");
  });
  test("the explicit Intel route uses genuine compiler-ID probes and direct LLD/llvm-lib rules", () => {
    const d = windowsDescriptor(), host = {cc: "/intel/bin/icx-cl", cxx: "/intel/bin/icx-cl"};
    const p = windowsJitToolchainPlan(d, "/sdk", "/native", "/output", host), rules = p.ruleOverrideText;
    assert.equal(p.ruleOverride, "/output/intel-link-rules.cmake"); assert.ok(rules);
    assert.equal(p.compilerTools.CMAKE_LINKER, d.tools.lldLink); assert.equal(p.compilerTools.CMAKE_AR, "/output/bin/llvm-lib");
    assert.match(p.toolchain, /CMAKE_USER_MAKE_RULES_OVERRIDE_C /); assert.match(p.toolchain, /CMAKE_USER_MAKE_RULES_OVERRIDE_CXX /);
    assert.ok(rules.includes("/out:<TARGET>")); assert.doesNotMatch(rules, /\/OUT:<TARGET>/);
    assert.match(rules, /CMAKE_C_LINK_EXECUTABLE \[==\[<CMAKE_LINKER>[^\n]+<CMAKE_C_LINK_FLAGS>/);
    assert.match(rules, /CMAKE_CXX_LINK_EXECUTABLE \[==\[<CMAKE_LINKER>[^\n]+<CMAKE_CXX_LINK_FLAGS>/);
    assert.match(rules, /CMAKE_C_CREATE_STATIC_LIBRARY \[==\[<CMAKE_AR>[^\n]*<OBJECTS>/);
    assert.match(rules, /CMAKE_CXX_CREATE_STATIC_LIBRARY \[==\[<CMAKE_AR>[^\n]*<OBJECTS>/);
    for (const language of ["C", "CXX"]) {
      const link = rules.split("\n").find(line => line.startsWith(`set(CMAKE_${language}_LINK_EXECUTABLE `));
      assert.ok(link?.includes('"/os/lib/libcmt.lib" "/os/lib/libucrt.lib" "/os/lib/kernel32.lib"'));
      assert.ok(rules.includes(`set(CMAKE_${language}_STANDARD_LIBRARIES_INIT [==[]==])`));
    }
    assert.doesNotMatch(rules, /string\(APPEND|MOM_INTEL_RUNTIME_LIBRARIES_ADDED/);
    assert.match(rules, /set\(CMAKE_EXE_LINKER_FLAGS_INIT \[==\[\/libpath:/);
    assert.match(rules, /set\(CMAKE_C_LINKER_WRAPPER_FLAG \[==\[\]==\]/);
    assert.doesNotMatch(rules + p.toolchain, /set\s*\(\s*CMAKE_(?:C|CXX)_COMPILER_ID|SKIP_(?:C|CXX)_COMPILER|CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY|Qfreestanding/i);
    assert.match(p.toolchain, /set\(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE\)/);
    assert.throws(() => windowsIntelRuleOverride("/libpath:/sdk;set(BAD)", []), /literal flags/);
    assert.throws(() => windowsIntelRuleOverride("/libpath:/sdk", ["/sdk/libcmt.lib;set(BAD)"]), /literal flags/);
  });
  test("shared Windows rules preserve exact archive paths and case without basename guessing", () => {
    const files = ["/sdk one/Kernel32.Lib", "/sdk two/kernel32.lib", "/sdk one/libcmt.lib"];
    const rules = windowsIntelRuleOverride("/machine:x64", files);
    const exact = files.map(file => '"' + file + '"').join(" ");
    for (const language of ["C", "CXX"]) {
      const link = rules.split("\n").find(line => line.startsWith(`set(CMAKE_${language}_LINK_EXECUTABLE `));
      assert.ok(link?.includes(exact), "ABI detection must retain libraries even when CMake clears STANDARD_LIBRARIES");
    }
    assert.throws(() => windowsIntelRuleOverride("/machine:x64", ["/sdk/library.a"]), /literal flags/);
  });
  test("uses original Clang23 and native LLVM archive aliases", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.equal(p.compilerTools.CMAKE_CXX_COMPILER, "/sdk/bin/clang-cl"); assert.equal(p.compilerTools.CMAKE_AR, "/new/bin/llvm-lib");
    assert.equal(p.compilerTools.CMAKE_MT, "/tools/llvm-mt", "the unmodified Clang/MSVC toolchain retains its manifest tool");
    assert.equal(p.compilerTools.CMAKE_RANLIB, "/new/bin/llvm-ranlib"); assert.doesNotMatch(p.toolchain, /wrapper|\.sh|CMAKE_USER_MAKE_RULES_OVERRIDE/);
    assert.equal(p.sourceOptionsFile, "/new/heatutils-source-options.cmake");
    assert.ok(p.toolchain.includes("CMAKE_PROJECT_TOP_LEVEL_INCLUDES [==[/new/heatutils-source-options.cmake]==]"));
    assert.match(p.sourceOptionsText, /GLOBAL PROPERTY IN_TRY_COMPILE/);
    assert.match(p.sourceOptionsText, /CMAKE_SOURCE_DIR\}\/lib\/Analysis\/HeatUtils\.cpp/);
    assert.match(p.sourceOptionsText, /DIRECTORY "\$\{CMAKE_SOURCE_DIR\}\/lib\/Analysis" APPEND PROPERTY COMPILE_OPTIONS \[==\[-fno-builtin\]==\]/);
    assert.doesNotMatch(p.sourceOptionsText, /CMAKE_(?:C|CXX)_FLAGS|add_compile_options/);
    assert.equal(p.ruleOverride, undefined); assert.equal(p.ruleOverrideText, undefined);
  });
  test("keeps MSVC host ABI and the application's static CRT", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.match(p.toolchain, /CMAKE_SYSTEM_NAME Windows/); assert.match(p.toolchain, /--target=x86_64-pc-windows-msvc/);
    assert.match(p.toolchain, /CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded/); assert.doesNotMatch(p.toolchain, /\/MD\b/);
  });
  test("Clang builtin intrinsics precede Windows SDK headers", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.ok(p.toolchain.indexOf("-I/sdk/lib/clang/23/include") < p.toolchain.indexOf("-I/virtual/msvc"));
    assert.match(p.toolchain, /-nostdinc/);
  });
  test("selects existing original host TableGen and translator binaries", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.equal(p.compilerTools.LLVM_TABLEGEN, "/native/llvm-tblgen"); assert.equal(p.compilerTools.CLANG_TABLEGEN, "/native/clang-tblgen");
    assert.equal(p.compilerTools.LLVM_SPIRV, "/sdk/bin/llvm-spirv");
  });
  test("MASM uses the selected LLVM tool directly in 64-bit mode", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.equal(p.compilerTools.CMAKE_ASM_MASM_COMPILER, "/tools/llvm-ml"); assert.match(p.toolchain, /CMAKE_ASM_MASM_FLAGS_INIT --m64/);
  });
  test("paths with spaces remain individual CMake/compiler literals", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk path", "/native path", "/new path");
    assert.ok(p.toolchain.includes("[==[/sdk path/bin/clang-cl]==]")); assert.ok(p.toolchain.includes('"-I/sdk path/lib/clang/23/include"'));
  });
  test("interpolation and list injection cannot modify the cross recipe", () => {
    for (const sdk of ["relative", "/bad;list", '/bad"path', "/bad]==]path", "/bad\npath", "/bad" + String.fromCharCode(36) + "{OTHER}"]) {
      assert.throws(() => windowsJitToolchainPlan(windowsDescriptor(), sdk, "/native", "/new"), /literal/);
    }
  });
  test("host include/library discovery cannot leak into the Windows build", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    for (const name of ["LIBRARY", "INCLUDE", "PACKAGE"]) {assert.match(p.toolchain, new RegExp("CMAKE_FIND_ROOT_PATH_MODE_" + name + " ONLY"));}
  });
  test("Windows cross probes link to verify target runtime symbols and threads", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.match(p.toolchain, /set\(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE\)/);
    assert.doesNotMatch(p.toolchain, /set\(CMAKE_HAVE_(?:LIBC_PTHREAD|PTHREADS_CREATE|PTHREAD_CREATE) FALSE CACHE BOOL/);
    assert.doesNotMatch(p.toolchain, /set\(THREADS_HAVE_PTHREAD_ARG FALSE CACHE BOOL/);
  });
  test("the generated VFS is separate from the unchanged application SDK overlay", () => {
    const p = windowsJitToolchainPlan(windowsDescriptor(), "/sdk", "/native", "/new");
    assert.equal(p.config.windowsSdk.overlay, "/vfs.json"); assert.equal(p.overlay, "/new/vfs.json"); assert.ok(p.toolchain.includes('"/new/vfs.json"'));
  });
  test("ordinary system-DLL restrictions are reused, not relaxed", () => {
    const d = windowsDescriptor(); d.windowsSdk.allowedImports.push("node.dll");
    assert.throws(() => windowsJitToolchainPlan(d, "/sdk", "/native", "/new"), /ordinary system DLLs/);
  });
  test("toolchain preparation never starts implicitly", () => {
    assert.throws(() => prepareWindowsJitToolchain("/config", "/sdk", "/native", "/new", {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("preparation freezes selected headers and creates aliases without running tools", t => {
    const f = crossFixture(t), hostCompiler = path.join(f.root, "intel/icx-cl");
    fs.mkdirSync(path.dirname(hostCompiler), {recursive: true}); fs.writeFileSync(hostCompiler, "fixture compiler");
    const host = {cc: hostCompiler, cxx: hostCompiler};
    const r = prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, f.output, {MOM_AOT_BUILD_ONLY: "1"}, host);
    assert.equal(r.compiled, false); assert.equal(r.executed, false); assert.equal(r.inputs[f.header]?.length, 64); assert.equal(r.inputs["/nonexistent.h"], undefined);
    assert.equal(fs.realpathSync(path.join(f.output, "bin/llvm-lib")), path.join(f.sdk, "bin/llvm-ar"));
    assert.equal(fs.realpathSync(path.join(f.output, "lib/kernel32.lib")), f.library);
    const rules = path.join(f.output, "intel-link-rules.cmake");
    assert.equal(r.inputs[rules], fileHash(rules), "generated Intel link/archive rules are frozen in toolchain provenance");
    const sourceOptions = path.join(f.output, "heatutils-source-options.cmake");
    assert.equal(r.inputs[sourceOptions], fileHash(sourceOptions), "the Windows-only source hook is frozen in toolchain provenance");
    const windowsConfig = {version: 1, target: "windows", sources: "/sources.json", sdk: f.sdk, resources: "/resources",
      tools: {cmake: "/cmake", ninja: "/ninja", toolchain: r.toolchain}, jobs: 2, inputs: Object.keys(r.inputs)};
    assert.ok(jitSdkConfig(windowsConfig).inputs?.includes(sourceOptions), "the existing Windows config input list can carry the generated hook hash");
    const vfs = JSON.parse(fs.readFileSync(path.join(f.output, "vfs.json"), "utf8")); assert.equal(vfs["use-external-names"], true); assert.equal(vfs.roots.length, 1);
    assert.throws(() => prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, f.output, {MOM_AOT_BUILD_ONLY: "1"}), /new canonical/);
  });
  test("CMake applies the hash-bound HeatUtils option only to Windows IntelLLVM", t => {
    const cmakeVersion = spawnSync("cmake", ["--version"], {encoding: "utf8", timeout: 10000});
    if (cmakeVersion.error || cmakeVersion.status !== 0 || !/^cmake version (?:3\.(?:2[4-9]|[3-9]\d)|[4-9]\.\d+)/m.test(cmakeVersion.stdout || "")) {
      return t.skip("CMake 3.24+ is required to exercise CMAKE_PROJECT_TOP_LEVEL_INCLUDES");
    }
    const f = crossFixture(t), toolchain = prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, f.output,
        {MOM_AOT_BUILD_ONLY: "1"}), source = path.join(f.root, "heatutils-source"), analysis = path.join(source, "lib/Analysis");
    fs.mkdirSync(analysis, {recursive: true});
    fs.writeFileSync(path.join(source, "CMakeLists.txt"), `cmake_minimum_required(VERSION 3.24)
project(HeatUtilsFlagFixture LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(WIN32 \${MOM_TEST_WINDOWS})
set(CMAKE_CXX_COMPILER_ID \${MOM_TEST_COMPILER_ID})
add_subdirectory(${path.relative(source, analysis)})
`);
    const heatUtils = path.join(analysis, "HeatUtils.cpp");
    fs.writeFileSync(path.join(analysis, "CMakeLists.txt"), [
      `if(EXISTS "${heatUtils}")`,
      "  add_library(heat_utils OBJECT HeatUtils.cpp)",
      "endif()",
      "add_library(other OBJECT Other.cpp)",
      "",
    ].join("\n"));
    fs.writeFileSync(heatUtils, "int heat_utils() { return 0; }\n");
    fs.writeFileSync(path.join(analysis, "Other.cpp"), "int other() { return 0; }\n");
    const modes = /** @type {Array<[string, string, string, boolean]>} */ ([["windows-intel", "ON", "IntelLLVM", true], ["windows-clang", "ON", "Clang", false],
      ["linux-intel", "OFF", "IntelLLVM", false]]);
    for (const [name, windows, compiler, expected] of modes) {
      const build = path.join(f.root, name), args = ["-S", source, "-B", build, "-G", "Ninja",
        "-DCMAKE_TOOLCHAIN_FILE=" + toolchain.toolchain, "-DCMAKE_CXX_COMPILER_FORCED=TRUE", "-DCMAKE_CXX_COMPILER_WORKS=TRUE",
        "-DCMAKE_CXX_COMPILER_ID=" + compiler, "-DCMAKE_CXX_COMPILER_VERSION=2026.2.0",
        "-DCMAKE_CXX_COMPILER_FRONTEND_VARIANT=MSVC", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DMOM_TEST_WINDOWS=" + windows, "-DMOM_TEST_COMPILER_ID=" + compiler];
      const result = spawnSync("cmake", args, {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
      assert.equal(result.status, 0, `${name} CMake configure failed:\n${result.stdout}\n${result.stderr}`);
      const database = /** @type {Array<{file: string, command: string}>} */ (JSON.parse(fs.readFileSync(path.join(build, "compile_commands.json"), "utf8"))),
        heat = database.find(row => row.file.endsWith("/lib/Analysis/HeatUtils.cpp")), other = database.find(row => row.file.endsWith("/lib/Analysis/Other.cpp"));
      assert.ok(heat); assert.ok(other);
      assert.equal((heat.command.match(/(?:^|\s)-fno-builtin(?:\s|$)/g) || []).length, expected ? 1 : 0, `${name} HeatUtils option`);
      assert.doesNotMatch(other.command, /(?:^|\s)-fno-builtin(?:\s|$)/, `${name} unrelated source option`);
    }
    fs.unlinkSync(heatUtils);
    const missing = spawnSync("cmake", ["-S", source, "-B", path.join(f.root, "windows-intel-missing"), "-G", "Ninja",
      "-DCMAKE_TOOLCHAIN_FILE=" + toolchain.toolchain, "-DCMAKE_CXX_COMPILER_FORCED=TRUE", "-DCMAKE_CXX_COMPILER_WORKS=TRUE",
      "-DCMAKE_CXX_COMPILER_ID=IntelLLVM", "-DCMAKE_CXX_COMPILER_VERSION=2026.2.0",
      "-DCMAKE_CXX_COMPILER_FRONTEND_VARIANT=MSVC", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
      "-DMOM_TEST_WINDOWS=ON", "-DMOM_TEST_COMPILER_ID=IntelLLVM"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    assert.notEqual(missing.status, 0, "a Windows IntelLLVM configure must reject a missing main LLVM target source");
    assert.match(missing.stderr, /missing lib\/Analysis\/HeatUtils\.cpp/i);
    const generatedHook = path.join(f.output, "heatutils-source-options.cmake");
    assert.equal(toolchain.inputs[generatedHook], fileHash(generatedHook));
    assert.equal(toolchain.inputs[toolchain.toolchain], fileHash(toolchain.toolchain));
  });
  test("case-colliding Windows archives cannot select conflicting bytes", t => {
    const f = crossFixture(t); fs.writeFileSync(path.join(f.root, "os/lib/kernel32.lib"), "different bytes");
    assert.throws(() => prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, f.output, {MOM_AOT_BUILD_ONLY: "1"}), /Conflicting/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("an output inside its compiler SDK is rejected before mutation", t => {
    const f = crossFixture(t), output = path.join(f.sdk, "new");
    assert.throws(() => prepareWindowsJitToolchain(f.descriptor, f.sdk, f.native, output, {MOM_AOT_BUILD_ONLY: "1"}), /overlaps/);
    assert.equal(fs.existsSync(output), false);
  });
});
