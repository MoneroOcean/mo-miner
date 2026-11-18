"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {windowsSourceJitInputs, windowsStaticSdkPlan, buildWindowsStaticSdk} = require("../scripts/build-native-windows-sdk");
const {sdkComponents, verifyPreparedSdkSources} = require("../scripts/prepare-native-sdk-sources");
const components = sdkComponents("windows");
const {windowsToolEnvironment} = require("../scripts/build-native-windows-controller");
const {dpcppEntries} = require("../scripts/native-compiler-isolation"), {fileHash} = require("../scripts/build-source-inputs");
const {spawnSync} = require("node:child_process");
const {windowsIntelRuleOverride} = require("../scripts/prepare-native-jit-toolchain");

function fixture() {
  const root = "/sdk fixture/space", file = (/** @type {string} */ name) => path.join(root, name);
  const config = {version: 1, tools: {clangCl: file("bin/clang-cl"), lldLink: file("bin/lld-link"), shermaot: file("bin/shermes"), llvmNm: file("bin/llvm-nm"), llvmReadobj: file("bin/llvm-readobj"), clangResourceInclude: file("resource/include")},
    hermes: {source: file("hermes"), build: file("hermes-build"), shermaot: file("bin/shermes")}, windowsSdk: {overlay: file("overlay.json"), sourceRoot: file("os"), includeRoots: [file("include")], libraryRoots: [file("lib")],
      msvcLibraries: [file("lib/libcmt.lib")], ucrtLibraries: [file("lib/libucrt.lib")], importLibraries: [file("lib/Kernel32.Lib")], allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: file("openssl/include"), ssl: file("ssl.lib"), crypto: file("crypto.lib"), libuv: file("uv.lib"), zlib: file("z.lib")}};
  const roots = Object.fromEntries(Object.keys(components).map(name => [name, file("source/" + name)]));
  return {config, roots};
}

test.describe("Native Windows static SDK build recipe", {skip: process.platform !== "linux" && "Linux cross-build recipe"}, () => {
  test("the plan builds all original runtime and adapter targets, not a tiny replacement", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new output", 2);
    for (const name of ["mom-static-dpcpp-libs", "ur_adapter_cuda", "ur_adapter_level_zero", "ur_adapter_level_zero_v2", "ur_adapter_opencl", "umf", "hwloc"]) {assert.ok(plan.targets.includes(name));}
    assert.ok(plan.artifacts.includes("lib/level_zero_utils.lib")); assert.ok(plan.artifacts.includes("opencl-icd-loader/OpenCL.lib"));
    assert.deepEqual(plan.compile.slice(-3), ["--", "-k", "0"]);
  });
  test("cross compilation uses the explicit tools and static CRT without host-ABI defaults", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new output", 2);
    assert.match(plan.toolchain, /CMAKE_SYSTEM_NAME Windows/); assert.match(plan.toolchain, /CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded/);
    assert.ok(plan.toolchain.includes(f.config.tools.clangCl)); assert.match(plan.toolchain, /--target=x86_64-pc-windows-msvc/);
    assert.match(plan.toolchain, /-nostdinc/); assert.doesNotMatch(plan.toolchain, /\/MD|native-cuda-adapter\.js|\/home\/sap\/cache/);
  });
  test("Intel host compiler selection leaves auxiliary LLVM tools on their explicit SDK", () => {
    const f = fixture(); f.config.tools.clangCl = "/intel/bin/icx-cl";
    const plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new output", 2);
    assert.equal(plan.compilerTools.CMAKE_CXX_COMPILER, "/intel/bin/icx-cl");
    for (const [key, name] of Object.entries({CMAKE_RC_COMPILER: "llvm-rc", CMAKE_MT: "llvm-mt", CMAKE_AR: "llvm-lib", CMAKE_RANLIB: "llvm-ranlib"})) {
      assert.equal(Reflect.get(plan.compilerTools, key), path.join(path.dirname(f.config.tools.llvmReadobj), name));
    }
    assert.doesNotMatch(plan.toolchain, /\/intel\/bin\/llvm-/);
  });
  test("runtime and compiler SDKs share the same Intel archive/link rules", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new output", 2);
    assert.equal(plan.ruleOverride, "/new output/intel-link-rules.cmake");
    assert.equal(plan.ruleOverrideText, windowsIntelRuleOverride(plan.definitions.CMAKE_EXE_LINKER_FLAGS,
      [...f.config.windowsSdk.msvcLibraries, ...f.config.windowsSdk.ucrtLibraries, ...f.config.windowsSdk.importLibraries]));
    assert.ok(plan.toolchain.includes("CMAKE_USER_MAKE_RULES_OVERRIDE_C "));
    assert.ok(plan.toolchain.includes("CMAKE_USER_MAKE_RULES_OVERRIDE_CXX "));
  });
  test("shared rules activate only for a real Intel compiler identity", t => {
    const probe = spawnSync("cmake", ["--version"], {encoding: "utf8"});
    if (probe.status !== 0) {return t.skip("CMake unavailable for compiler-free rules fixture");}
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-rules-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const rules = path.join(root, "rules.cmake"), script = path.join(root, "test.cmake");
    fs.writeFileSync(rules, windowsIntelRuleOverride("/machine:x64", ["/sdk/libcmt.lib"]));
    // Only exercise the conditional rule text; this is not a compiler-ID or SDK ABI probe.
    fs.writeFileSync(script, [
      'set(CMAKE_C_COMPILER_ID "Clang")', 'set(CMAKE_CXX_COMPILER_ID "Clang")',
      'set(CMAKE_C_LINK_EXECUTABLE "original")', `include("${rules.replaceAll("\\", "/")}")`,
      'if(NOT CMAKE_C_LINK_EXECUTABLE STREQUAL "original")', 'message(FATAL_ERROR "Non-Intel rules changed")', "endif()",
      'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")', `include("${rules.replaceAll("\\", "/")}")`,
      'if(NOT CMAKE_CXX_LINK_EXECUTABLE MATCHES "<CMAKE_LINKER>")', 'message(FATAL_ERROR "Intel linker not selected")', "endif()",
      // CMake clears these for its ABI probe; selected CRT archives must survive in the rule.
      'set(CMAKE_CXX_STANDARD_LIBRARIES "")',
      'string(FIND "${CMAKE_CXX_LINK_EXECUTABLE}" "/sdk/libcmt.lib" selected_crt)', // eslint-disable-line no-template-curly-in-string -- Literal CMake expansion.
      "if(selected_crt LESS 0)", 'message(FATAL_ERROR "ABI probe lost the selected CRT")', "endif()",
    ].join("\n"));
    const result = spawnSync("cmake", ["-P", script], {encoding: "utf8"});
    assert.equal(result.status, 0, result.stdout + result.stderr);
  });
  test("paths with spaces stay literal CMake and compiler tokens", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp path", "/cuda path", "/new output", 2);
    assert.ok(plan.toolchain.includes("[==[/sdk fixture/space/bin/clang-cl]==]"));
    assert.ok(plan.toolchain.includes('"-I/cuda path/include"')); assert.ok(plan.configure.includes("-B")); assert.ok(plan.configure.includes("/new output/build"));
  });
  test("CMake interpolation, list separators and control characters cannot alter selected roots", () => {
    const f = fixture();
    for (const root of ["relative", "/bad;extra", "/bad" + String.fromCharCode(36) + "{OTHER}", "/bad\nroot", '/bad"root', "/bad\\root"]) {
      assert.throws(() => windowsStaticSdkPlan(f.roots, f.config, root, "/cuda", "/new", 2), /literal absolute/);
    }
  });
  test("parallelism remains explicit and bounded", () => {
    const f = fixture();
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", jobs), /jobs/);}
    for (const jobs of [1, 2, 16]) {assert.ok(windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", jobs).compile.includes(String(jobs)));}
  });
  test("the existing OS-only descriptor policy rejects application-DLL allowances", () => {
    const f = fixture(); f.config.windowsSdk.allowedImports.push("node.dll");
    assert.throws(() => windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", 2), /ordinary system DLLs/);
  });
  test("all prepared source roots remain explicit in the actual configure command", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", 2);
    for (const root of Object.values(f.roots)) {assert.ok(plan.configure.some(arg => arg.endsWith("=" + root)));}
    assert.ok(plan.configure.includes("-DMOM_DPCPP_CUDA_SDK=/cuda"));
    assert.ok(plan.configure.includes("-DMOM_DPCPP_SDK=/dpcpp"));
  });
  test("shared environment admission strips ABI overrides without changing the parent", () => {
    const before = {PATH: "/bin", INCLUDE: "bad", LIB: "bad", CL: "/MD", CXXFLAGS: "bad"}, env = windowsToolEnvironment(before);
    assert.deepEqual(env, {PATH: "/bin"}); assert.equal(before.CL, "/MD");
  });
  test("a build cannot mutate outputs without explicit build-only admission", () => {
    assert.throws(() => buildWindowsStaticSdk("/missing", "/missing", "/missing", "/missing", "/missing", {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("an incomplete receipt cannot select an empty source-verification recipe", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sdk-receipt-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const file = path.join(root, "sources.json"), prepared = Object.fromEntries(Object.keys(components).map(name => [name, {}]));
    for (const value of [{}, {version: 1, status: "FAIL", prepared, compiled: false}, {version: 1, status: "PASS", prepared, compiled: false, recipe: {files: {}, graph: {}}}]) {
      fs.writeFileSync(file, JSON.stringify(value)); assert.throws(() => verifyPreparedSdkSources(file, "windows"), /source cohort|source inputs are stale/);
    }
  });
});

/** Shape-only archived build receipt. No original compiler or Windows execution claim.
 * @param {import('node:test').TestContext} t */
function jitFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-static-jit-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const repo = path.resolve(__dirname, ".."), file = path.join(root, "result.json"), map = path.join(root, "map.txt"); fs.writeFileSync(map, "map");
  const sourceHashes = Object.fromEntries(["scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "scripts/build-source-inputs.js"].map(name => {
    const selected = path.join(repo, name); return [selected, fileHash(selected)];
  }));
  const toolHashes = Object.fromEntries(["nm", "objcopy", "ar"].map(name => {const selected = path.join(root, name); fs.writeFileSync(selected, "tool"); return [selected, fileHash(selected)];}));
  fs.mkdirSync(path.join(root, "original")); fs.mkdirSync(path.join(root, "private"));
  const transformed = Array.from({length: 50}, (_, index) => {
    const name = index ? "leaf" + index + ".lib" : "sycl-jit.lib", input = path.join(root, "original", name), output = path.join(root, "private", name);
    fs.writeFileSync(input, "!<arch>\n"); fs.writeFileSync(output, "!<arch>\n");
    return {input, output, inputSha256: fileHash(input), sha256: fileHash(output)};
  });
  const checks = Object.fromEntries(["memberOrderPreserved", "nonDirectivePayloadsPreserved", "mappedDirectivesPreserved", "symbolOrderPreserved", "auxiliaryRecordsPreserved", "relocationsPreserved"].map(name => [name, true]));
  const receipt = {version: 1, prefix: "mom_dpcpp23_", entryPoints: [...dpcppEntries], buildOnly: true, executed: false, publicationQualified: false,
    transformed, map, mapSha256: fileHash(map), toolHashes, sourceHashes, checks};
  const save = () => fs.writeFileSync(file, JSON.stringify(receipt)); save();
  return {root, file, receipt, save};
}

test.describe("Windows original static source-JIT admission", {skip: process.platform !== "linux" && "Linux cross-build admission"}, () => {
  test("an omitted compiler stays explicitly disabled, never selected from an installed SDK", () => {
    assert.equal(windowsSourceJitInputs(undefined), null);
    const f = fixture(); assert.equal(Reflect.get(windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", 2).definitions, "MOM_DPCPP_JIT_ARCHIVE"), undefined);
  });
  test("a complete immutable compiler closure retains every original leaf", t => {
    const f = jitFixture(t), selected = windowsSourceJitInputs(f.file); assert.ok(selected);
    assert.equal(selected.libraries.length, 50); assert.equal(selected.archive, f.receipt.transformed[0]?.output);
    assert.equal(selected.files[f.file], fileHash(f.file)); assert.ok(Object.keys(selected.files).length >= 108);
  });
  test("the selected archive explicitly enables the original CMake host bridge", () => {
    const f = fixture(), plan = windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", 2, "/private/sycl-jit.lib");
    assert.ok(plan.configure.includes("-DMOM_DPCPP_JIT_ARCHIVE=/private/sycl-jit.lib"));
  });
  test("relative, shared, interpolated and wrong-OS compiler inputs are rejected", () => {
    const f = fixture();
    for (const file of ["relative.lib", "/bad.dll", "/bad.o", "/bad;extra.lib", "/bad\nfile.lib"]) {
      assert.throws(() => windowsStaticSdkPlan(f.roots, f.config, "/dpcpp", "/cuda", "/new", 2, file), /explicit isolated COFF archive/);
    }
  });
  test("partial and foreign ABI compiler receipts are refused before linking", t => {
    const f = jitFixture(t); f.receipt.entryPoints.pop(); f.save(); assert.throws(() => windowsSourceJitInputs(f.file), /Incomplete isolated/);
  });
  test("a truncated compiler graph cannot silently omit original dependencies", t => {
    const f = jitFixture(t); f.receipt.transformed.pop(); f.save(); assert.throws(() => windowsSourceJitInputs(f.file), /Incomplete isolated/);
  });
  test("missing relocation and payload preservation cannot qualify an input", t => {
    const f = jitFixture(t); f.receipt.checks["relocationsPreserved"] = false; f.save(); assert.throws(() => windowsSourceJitInputs(f.file), /Incomplete isolated/);
  });
  test("stale isolation source recipes must be rebuilt, not restamped", t => {
    const f = jitFixture(t), source = Object.keys(f.receipt.sourceHashes)[0]; assert.ok(source); f.receipt.sourceHashes[source] = "0".repeat(64); f.save();
    assert.throws(() => windowsSourceJitInputs(f.file), /input changed/);
  });
  test("tool metadata cannot overwrite a stale source-recipe hash", t => {
    const f = jitFixture(t), source = Object.keys(f.receipt.sourceHashes)[0], tool = Object.keys(f.receipt.toolHashes)[0]; assert.ok(source && tool);
    f.receipt.sourceHashes[source] = "0".repeat(64); delete f.receipt.toolHashes[tool]; f.receipt.toolHashes[source] = fileHash(source); f.save();
    assert.throws(() => windowsSourceJitInputs(f.file), /input changed/);
  });
  test("changed original or transformed archives invalidate the whole closure", t => {
    const f = jitFixture(t), original = f.receipt.transformed[0]; assert.ok(original); fs.appendFileSync(original.output, "changed");
    assert.throws(() => windowsSourceJitInputs(f.file), /input changed/);
  });
  test("an entry archive cannot be silently replaced by another compiler leaf", t => {
    const f = jitFixture(t); f.receipt.transformed.reverse(); f.save(); assert.throws(() => windowsSourceJitInputs(f.file), /must start/);
  });
  test("duplicate compiler archive identities are rejected", t => {
    const f = jitFixture(t), first = f.receipt.transformed[0]; assert.ok(first); f.receipt.transformed[1] = {...first}; f.save();
    assert.throws(() => windowsSourceJitInputs(f.file), /archive selection/);
  });
  test("changed linker maps and tool hashes cannot certify isolation", t => {
    const f = jitFixture(t); fs.appendFileSync(f.receipt.map, "changed"); assert.throws(() => windowsSourceJitInputs(f.file), /input changed/);
  });
  test("the Windows OpenCL binding is staged before the compiler sees its source path", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-opencl-binding-stage-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const sourceRoot = path.join(root, "source"), source = path.join(sourceRoot, "tests/reference/native-opencl-binding.cpp");
    const headers = path.join(root, "headers"), build = path.join(root, "build"), input = "int answer() { return 42; }\n";
    fs.mkdirSync(path.dirname(source), {recursive: true}); fs.mkdirSync(headers); fs.writeFileSync(source, input);
    const cmakeSource = fs.readFileSync(path.join(__dirname, "native/dpcpp-static/CMakeLists.txt"), "utf8");
    const blockStart = cmakeSource.indexOf("if(MOM_DPCPP_OPENCL_SOURCE)", cmakeSource.indexOf("add_custom_target(mom-static-dpcpp-libs"));
    const blockEnd = cmakeSource.indexOf("\n  endif()", blockStart);
    assert.ok(blockStart >= 0 && blockEnd > blockStart, "Windows OpenCL binding CMake block exists");
    const block = cmakeSource.slice(blockStart, blockEnd + "\n  endif()".length), dollar = String.fromCharCode(36);
    assert.match(block, /configure_file\("\$\{MOM_SOURCE_REPO\}\/tests\/reference\/native-opencl-binding\.cpp"\s+"\$\{CMAKE_CURRENT_BINARY_DIR\}\/native-opencl-binding\.cpp" COPYONLY\)/);
    assert.match(block, /add_executable\(native-opencl-binding-static "\$\{CMAKE_CURRENT_BINARY_DIR\}\/native-opencl-binding\.cpp"\)/);
    assert.match(block, /target_link_libraries\(native-opencl-binding-static PRIVATE ur_adapter_opencl OpenCL\)/);
    const cmakeLists = path.join(root, "CMakeLists.txt");
    fs.writeFileSync(cmakeLists, [
      "cmake_minimum_required(VERSION 3.20)", "project(staged_binding CXX)",
      "add_library(ur_adapter_opencl INTERFACE)", "add_library(OpenCL INTERFACE)",
      "add_custom_target(mom-static-dpcpp-libs)",
      "set(MOM_DPCPP_OPENCL_SOURCE TRUE)", `set(MOM_SOURCE_REPO "${sourceRoot.replaceAll("\\", "/")}")`,
      `set(MOM_DPCPP_OPENCL_HEADERS "${headers.replaceAll("\\", "/")}")`, block,
      "get_target_property(binding_sources native-opencl-binding-static SOURCES)",
      `if(NOT binding_sources STREQUAL "${dollar}{CMAKE_CURRENT_BINARY_DIR}/native-opencl-binding.cpp")`,
      '  message(FATAL_ERROR "binding target does not consume its staged source")', "endif()",
      "get_target_property(binding_links native-opencl-binding-static LINK_LIBRARIES)",
      'if(NOT binding_links STREQUAL "ur_adapter_opencl;OpenCL")',
      '  message(FATAL_ERROR "binding target no longer links the real adapter and OpenCL")', "endif()", "",
    ].join("\n"));
    const result = spawnSync("cmake", ["-S", root, "-B", build, "-G", "Ninja"], {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const staged = path.join(build, "native-opencl-binding.cpp"), buildNinja = fs.readFileSync(path.join(build, "build.ninja"), "utf8");
    assert.equal(fs.readFileSync(staged, "utf8"), input);
    const compileEdge = buildNinja.split("\n").find(line => line.startsWith("build ") && line.includes("native-opencl-binding-static.dir/"));
    assert.ok(compileEdge?.includes(staged), "Ninja compiles the build-tree copy");
    assert.ok(!compileEdge?.includes(source), "Ninja does not pass the root source path to the compiler");
  });
  test("the actual CMake import uses COFF archives only on Windows and rejects ambiguous inputs", t => {
    const f = jitFixture(t), cmakeSource = fs.readFileSync(path.join(__dirname, "native/dpcpp-static/CMakeLists.txt"), "utf8");
    const block = cmakeSource.slice(cmakeSource.indexOf("set(SYCL_ENABLE_EXTENSION_JIT OFF)"), cmakeSource.indexOf("set(SYCL_ENABLE_XPTI_TRACING OFF)"));
    assert.ok(block.length > 100); const archive = f.receipt.transformed[0]?.output; assert.ok(archive);
    const source = path.join(f.root, "CMakeLists.txt");
    const dollar = String.fromCharCode(36);
    fs.writeFileSync(source, ["cmake_minimum_required(VERSION 3.20)", "project(jit_import NONE)", block,
      "if(SYCL_ENABLE_EXTENSION_JIT)", "get_target_property(kind sycl-jit TYPE)", "message(STATUS \"JIT-ENABLED=" + dollar + "{SYCL_ENABLE_EXTENSION_JIT} KIND=" + dollar + "{kind}\")", "endif()", ""].join("\n"));
    const run = (/** @type {string} */ name, /** @type {string[]} */ flags) => spawnSync("cmake", ["-S", f.root, "-B", path.join(f.root, name), "-G", "Ninja", ...flags], {encoding: "utf8", timeout: 30000});
    const enabled = run("archive", ["-DWIN32=TRUE", "-DMOM_DPCPP_JIT_ARCHIVE=" + archive]); assert.equal(enabled.status, 0, enabled.stderr); assert.match(enabled.stdout, /JIT-ENABLED=ON KIND=STATIC_LIBRARY/);
    const object = run("object", ["-DWIN32=FALSE", "-DMOM_DPCPP_JIT_OBJECT=" + archive]); assert.equal(object.status, 0, object.stderr); assert.match(object.stdout, /KIND=OBJECT_LIBRARY/);
    const both = run("both", ["-DWIN32=TRUE", "-DMOM_DPCPP_JIT_ARCHIVE=" + archive, "-DMOM_DPCPP_JIT_OBJECT=" + archive]); assert.notEqual(both.status, 0); assert.match(both.stderr, /not both/);
    const wrong = run("wrong-os", ["-DWIN32=FALSE", "-DMOM_DPCPP_JIT_ARCHIVE=" + archive]); assert.notEqual(wrong.status, 0); assert.match(wrong.stderr, /explicit isolated compiler archive/);
  });
});
