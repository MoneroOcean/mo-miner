"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {createHash} = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {validateWindowsLlvmSdkConfig} = require("../scripts/build-native-windows-controller");
const {archiveSha256, buildPhases, windowsLlvmSdkPlan, llvmSourceSnapshot, verifyArchiveCatalog, llvmStaticArchiveTargets,
  verifyBuildPhases, verifyIntelCompilerIdentity, verifyCMakeCache, auditCompileCommands, verifyWindowsLlvmAdmission,
  windowsBuildInputSetsMatch, windowsLlvmPrehookPolicyUpgrade, windowsLlvmPrehookMetadataDrift, windowsLlvmPrehookConfiguredFilesMatch,
  windowsLlvmRefreshConfiguredFiles, windowsLlvmAdmissionOutputInventory,
  verifyWindowsReuseLayout, verifyWindowsReuseProvenance,
  verifyWindowsReuseSourceState, verifyWindowsReusableConfiguration,
  windowsLlvmConfiguredInputFiles, windowsLlvmConfiguredFiles, verifyAmdgpulibCallsCompileFlag,
  buildWindowsLlvmSdk, auditWindowsLlvmAdmission, refreshWindowsLlvmSdk} = require("../scripts/build-native-windows-llvm21-sdk");
const {fileHash, buildFileInputs} = require("../scripts/build-source-inputs");
const {jitBuildOutputs} = require("../scripts/native-jit-stages");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-llvm21-sdk-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const weirdInclude = path.join(root, "sdk]==]bracket", "include");
  const overlay = path.join(root, "windows sdk overlay.json"), header = path.join(weirdInclude, "windows.h");
  const descriptor = {version: 1,
    tools: {clangCl: path.join(root, "intel compiler", "icx-cl"), lldLink: path.join(root, "llvm", "lld-link"),
      shermaot: path.join(root, "shermes"), llvmNm: path.join(root, "llvm", "llvm-nm"), llvmReadobj: path.join(root, "llvm", "llvm-readobj"),
      clangResourceInclude: path.join(root, "clang resource", "include")},
    hermes: {source: path.join(root, "hermes source"), build: path.join(root, "hermes build"), shermaot: path.join(root, "shermes")},
    windowsSdk: {overlay, sourceRoot: path.join(root, "windows sdk"), includeRoots: [weirdInclude],
      libraryRoots: [path.join(root, "windows sdk", "lib")], msvcLibraries: [path.join(root, "windows sdk", "lib", "libcmt.lib")],
      ucrtLibraries: [path.join(root, "windows sdk", "lib", "libucrt.lib")], importLibraries: [path.join(root, "windows sdk", "lib", "kernel32.lib")],
      allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: path.join(root, "source include"), ssl: path.join(root, "ssl.lib"), crypto: path.join(root, "crypto.lib"),
      libuv: path.join(root, "uv.lib"), zlib: path.join(root, "zlib.lib")}};
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, roots: [
    {type: "file", name: header, "external-contents": header},
  ]}));
  const archive = path.join(root, "llvm-project-21.1.8.src.tar.xz");
  const output = path.join(root, "output with spaces");
  return {root, descriptor, overlay, archive, output};
}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

test.describe("Original Windows LLVM 21 host SDK producer", {skip: process.platform !== "linux" && "Linux cross-build recipe"}, () => {
  test("validates only consumed Windows LLVM tools and SDK roots while preserving full descriptors", t => {
    const f = fixture(t), tools = {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"},
      minimal = {version: 1,
        tools: {clangCl: f.descriptor.tools.clangCl, lldLink: f.descriptor.tools.lldLink,
          llvmNm: f.descriptor.tools.llvmNm, clangResourceInclude: f.descriptor.tools.clangResourceInclude},
        windowsSdk: Object.fromEntries(Object.entries(f.descriptor.windowsSdk).filter(([key]) => key !== "allowedImports"))},
      fullPlan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output, tools, 2),
      minimalPlan = windowsLlvmSdkPlan(minimal, f.archive, f.output, tools, 2);
    assert.deepEqual(minimalPlan, fullPlan, "the full controller descriptor maps to the same LLVM-only plan");
    assert.deepEqual(windowsLlvmSdkPlan({...f.descriptor, application: {core: path.join(f.root, "cpu-core.lib")}},
      f.archive, f.output, tools, 2), fullPlan, "unconsumed application metadata does not alter the LLVM-only plan");
    assert.deepEqual(validateWindowsLlvmSdkConfig(minimal), minimal,
      "unused controller fields and import allowlists are not required by the static LLVM producer");

    for (const field of ["clangCl", "lldLink", "llvmNm", "clangResourceInclude"]) {
      const invalid = JSON.parse(JSON.stringify(minimal)); delete invalid.tools[field];
      assert.throws(() => validateWindowsLlvmSdkConfig(invalid), /absolute tool/,
        `the LLVM producer still requires its consumed tool ${field}`);
    }
    for (const field of ["overlay", "sourceRoot", "includeRoots", "libraryRoots", "msvcLibraries", "ucrtLibraries", "importLibraries"]) {
      const invalid = JSON.parse(JSON.stringify(minimal)); delete invalid.windowsSdk[field];
      assert.throws(() => validateWindowsLlvmSdkConfig(invalid), /Windows SDK|MSVC libraries|UCRT libraries/,
        `the LLVM producer still requires its consumed SDK input ${field}`);
    }
  });

  test("constructs refresh generations around one validated source/build cache", t => {
    const f = fixture(t), cacheRoot = path.join(f.root, "cache-root"), source = path.join(cacheRoot, "source"), build = path.join(cacheRoot, "build"),
      descriptorFile = path.join(f.root, "descriptor.json"), archive = path.join(f.root, "llvm-project-21.1.8.src.tar.xz"),
      tools = {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, paths = {cacheRoot, source, build};
    fs.mkdirSync(cacheRoot, {mode: 0o700}); fs.mkdirSync(source, {mode: 0o755}); fs.mkdirSync(build, {mode: 0o755});
    const query = path.join(build, ".cmake", "api", "v1", "query", "codemodel-v2");
    fs.mkdirSync(path.dirname(query), {recursive: true}); fs.writeFileSync(query, "");
    const artifact = path.join(build, "lib", "fixture.lib"); fs.mkdirSync(path.dirname(artifact)); fs.writeFileSync(artifact, "archive\n");
    const target = {name: "fixture", project: "llvm", sourceDirectory: source, artifact, id: "fixture:STATIC_LIBRARY", sources: 1};
    const first = windowsLlvmSdkPlan(f.descriptor, archive, cacheRoot, tools, 1), latest = path.join(f.root, "generation-2");
    // LLVM's completed CMake graph uses /tmp source operands. A controller/JIT-only
    // filename fix must not invalidate that graph or force all compiler objects to rebuild.
    assert.doesNotMatch(first.ruleOverrideText, /CMAKE_(?:C|CXX)_COMPILE_OBJECT|\/T[cp]<SOURCE>/);
    fs.writeFileSync(first.overlay, first.overlayText); fs.writeFileSync(first.ruleOverride, first.ruleOverrideText);
    fs.writeFileSync(first.toolchainFile, first.toolchain);
    assert.throws(() => verifyWindowsReusableConfiguration(first), /source-options hook/,
      "a pre-hook cache must not be silently adopted or restamped");
    fs.writeFileSync(first.sourceOptionsFile, first.sourceOptionsText);
    verifyWindowsReuseLayout(cacheRoot, latest, first, descriptorFile, paths, [target], {});
    fs.mkdirSync(latest, {mode: 0o700});
    const second = windowsLlvmSdkPlan(f.descriptor, archive, latest, tools, 1, paths), next = path.join(f.root, "generation-3");
    assert.equal(second.source, source); assert.equal(second.build, build); assert.equal(second.reuse?.cacheRoot, cacheRoot);
    assert.equal(second.overlay, first.overlay); assert.equal(second.toolchainFile, first.toolchainFile);
    assert.equal(second.ruleOverride, first.ruleOverride); assert.deepEqual(second.configure, first.configure);
    assert.equal(second.overlayText, first.overlayText); assert.equal(second.toolchain, first.toolchain);
    assert.doesNotThrow(() => verifyWindowsReusableConfiguration(second));
    // The real Windows header overlay is about 700 KiB, unlike these small fixtures.
    const largeOverlay = {...second, overlayText: "x".repeat(700000)};
    fs.writeFileSync(second.overlay, largeOverlay.overlayText);
    assert.doesNotThrow(() => verifyWindowsReusableConfiguration(largeOverlay));
    fs.appendFileSync(second.overlay, "growth");
    assert.throws(() => verifyWindowsReusableConfiguration(largeOverlay), /configure inputs changed/);
    fs.writeFileSync(second.overlay, second.overlayText);
    const ordinary = windowsLlvmSdkPlan(f.descriptor, archive, next, tools, 1);
    assert.notDeepEqual(ordinary.configure, second.configure, "an ordinary output still owns its generated config paths");
    verifyWindowsReuseLayout(latest, next, second, descriptorFile, paths, [target], {});
    assert.equal(fs.existsSync(next), false, "layout validation does not create or mutate the new generation");
    assert.throws(() => verifyWindowsReuseLayout(latest, next, second, descriptorFile,
      {...paths, source: path.join(f.root, "wrong-source")}, [target], {}), /canonical source\/build cache/);

    const outside = path.join(f.root, "outside-state"); fs.writeFileSync(outside, "state\n");
    fs.symlinkSync(outside, path.join(build, ".ninja_log"));
    assert.throws(() => verifyWindowsReuseLayout(latest, next, second, descriptorFile, paths, [target], {}), /regular in-tree file/);
    fs.unlinkSync(path.join(build, ".ninja_log"));
    const outsideArchive = path.join(f.root, "outside.lib"); fs.writeFileSync(outsideArchive, "archive\n");
    fs.unlinkSync(artifact); fs.symlinkSync(outsideArchive, artifact);
    assert.throws(() => verifyWindowsReuseLayout(latest, next, second, descriptorFile, paths, [target], {}), /regular in-tree file/);
    fs.writeFileSync(first.toolchainFile, first.toolchain + "# drift\n");
    assert.throws(() => verifyWindowsReusableConfiguration(second), /full new configure\/build/);
  });

  test("permits only fixed recipe-file drift and binds refresh ancestry/source state", t => {
    const f = fixture(t), recipe = path.join(__dirname, "../scripts/build-native-windows-llvm21-sdk.js"), other = path.join(f.root, "input.lib"),
      expected = {[recipe]: "a".repeat(64), [other]: "b".repeat(64)}, changedRecipe = {...expected, [recipe]: "c".repeat(64)};
    assert.equal(windowsBuildInputSetsMatch(expected, changedRecipe, true), true);
    assert.equal(windowsBuildInputSetsMatch(expected, changedRecipe), false);
    assert.equal(windowsBuildInputSetsMatch(expected, {...changedRecipe, [other]: "d".repeat(64)}, true), false);
    assert.equal(windowsBuildInputSetsMatch(expected, {[recipe]: changedRecipe[recipe] || ""}, true), false);
    const metadata = path.join(f.root, "prior-output"); fs.mkdirSync(metadata, {mode: 0o700});
    const files = {result: "result.json", current: "result-current.json", admission: "admitted.json", objects: "admitted-objects.json"};
    for (const name of Object.values(files)) {fs.writeFileSync(path.join(metadata, name), name + "\n");}
    const provenance = {from: metadata, priorResultSha256: fileHash(path.join(metadata, files.result)),
      priorCurrentSha256: fileHash(path.join(metadata, files.current)), priorAdmissionSha256: fileHash(path.join(metadata, files.admission)),
      priorObjectsSha256: fileHash(path.join(metadata, files.objects))};
    assert.doesNotThrow(() => verifyWindowsReuseProvenance(provenance));
    fs.appendFileSync(path.join(metadata, files.current), "tamper\n");
    assert.throws(() => verifyWindowsReuseProvenance(provenance), /metadata changed/);
    fs.writeFileSync(path.join(metadata, files.current), files.current + "\n");
    fs.appendFileSync(path.join(metadata, files.objects), "tamper\n");
    assert.throws(() => verifyWindowsReuseProvenance(provenance), /metadata changed/);
    const sourceState = {sha256: "a".repeat(64), files: 1, bytes: 1};
    assert.doesNotThrow(() => verifyWindowsReuseSourceState(sourceState, {...sourceState}));
    assert.throws(() => verifyWindowsReuseSourceState(sourceState, {...sourceState, bytes: 2}), /source changed/);
  });

  test("recognizes only the exact pre-hook configure policy for refresh", t => {
    const f = fixture(t), tools = {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"},
      plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output, tools, 4), descriptorFile = path.join(f.root, "descriptor.json"),
      prior = {version: 1, role: plan.role, configure: plan.configure.filter(argument =>
        argument !== "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + plan.sourceOptionsFile),
      plan: {descriptorFile, archive: f.archive, source: plan.source, output: f.output, tools, jobs: 4}};
    assert.equal(prior.configure.length, 47); assert.equal(plan.configure.length, 48);
    assert.equal(plan.configure.length, prior.configure.length + 1, "refresh adds only the source-options hook definition");
    assert.equal(windowsLlvmPrehookPolicyUpgrade(prior, plan), true);
    assert.equal(windowsLlvmPrehookPolicyUpgrade({...prior, configure: prior.configure.map((argument, index) =>
      index === 8 ? argument + "-changed" : argument)}, plan), false, "any other configure argv drift is rejected");
    assert.equal(windowsLlvmPrehookPolicyUpgrade({...prior, plan: {...prior.plan, tools: {...tools, cmake: "/tmp/other-cmake"}}}, plan),
      false, "toolchain identity drift is rejected");
    assert.equal(windowsLlvmPrehookPolicyUpgrade({...prior, plan: {...prior.plan, source: path.join(f.root, "other-source")}}, plan),
      false, "source-root drift is rejected");
    assert.equal(windowsLlvmPrehookPolicyUpgrade({...prior, plan: {...prior.plan, cacheRoot: f.root}}, plan), false,
      "unknown legacy plan fields are rejected");
    assert.equal(windowsLlvmPrehookPolicyUpgrade({...prior, sourceOptions: {path: plan.sourceOptionsFile, sha256: "a".repeat(64)}}, plan),
      false, "a claimed source-options proof is not treated as a pre-hook receipt");
  });

  test("pre-hook metadata adoption permits only cache/log bytes and a same-content timestamped File API rename", () => {
    const oldIndex = ".cmake/api/v1/reply/index-2026-11-04T19-55-22-0131.json",
      newIndex = ".cmake/api/v1/reply/index-2026-11-05T23-37-49-0006.json",
      expected = {"CMakeCache.txt": "mode:644:sha256:" + "a".repeat(64), ".ninja_log": "mode:644:sha256:" + "b".repeat(64),
        [oldIndex]: "mode:644:sha256:" + "c".repeat(64), "build.ninja": "mode:644:sha256:" + "d".repeat(64),
        "CMakeFiles/rules.ninja": "mode:644:sha256:" + "e".repeat(64), "lib/fixture.obj": "mode:644:sha256:" + "f".repeat(64)},
      actual = /** @type {Record<string, string>} */ ({...expected, "CMakeCache.txt": "mode:644:sha256:" + "1".repeat(64),
        ".ninja_log": "mode:644:sha256:" + "2".repeat(64), [newIndex]: expected[oldIndex]});
    delete actual[oldIndex];
    const drift = windowsLlvmPrehookMetadataDrift(expected, actual);
    assert.ok(drift);
    assert.equal(drift?.length, 4);
    assert.deepEqual(Object.fromEntries(drift.map(row => [row.path, [row.before, row.after]])), {
      [oldIndex]: [expected[oldIndex], null], [newIndex]: [null, expected[oldIndex]],
      ".ninja_log": [expected[".ninja_log"], actual[".ninja_log"]],
      "CMakeCache.txt": [expected["CMakeCache.txt"], actual["CMakeCache.txt"]],
    });
    assert.equal(windowsLlvmPrehookMetadataDrift(expected, expected), undefined);
    assert.throws(() => windowsLlvmPrehookMetadataDrift(expected, {...actual, "lib/fixture.obj": "mode:644:sha256:" + "9".repeat(64)}),
      /semantic build output changed/);
    assert.throws(() => windowsLlvmPrehookMetadataDrift(expected, {...actual, "CMakeCache.txt": "mode:600:sha256:" + "1".repeat(64)}),
      /state-file mode/);
    const noLog = /** @type {Record<string, string>} */ ({...actual}); delete noLog[".ninja_log"];
    assert.throws(() => windowsLlvmPrehookMetadataDrift(expected, noLog), /state-file mode/);
    assert.throws(() => windowsLlvmPrehookMetadataDrift(expected, {...actual, [newIndex]: "mode:644:sha256:" + "9".repeat(64)}),
      /File API index changed/);
    assert.throws(() => windowsLlvmPrehookMetadataDrift(expected, {...actual,
      ".cmake/api/v1/reply/index-2026-11-06T01-02-03-0004.json": expected[oldIndex]}), /single CMake File API generation/);
  });

  test("pre-hook configured graph comparison allows only CMakeCache content drift", () => {
    const build = "/private/cache/build", cache = path.join(build, "CMakeCache.txt"),
      expected = {[cache]: "a".repeat(64), [path.join(build, "build.ninja")]: "b".repeat(64),
        [path.join(build, "compile_commands.json")]: "c".repeat(64)},
      cacheChanged = {...expected, [cache]: "d".repeat(64)};
    assert.equal(windowsLlvmPrehookConfiguredFilesMatch(expected, cacheChanged, build), true);
    assert.equal(windowsLlvmPrehookConfiguredFilesMatch(expected, {...cacheChanged,
      [path.join(build, "build.ninja")]: "e".repeat(64)}, build), false);
    const missingCache = /** @type {Record<string, string>} */ ({...cacheChanged}); delete missingCache[cache];
    assert.equal(windowsLlvmPrehookConfiguredFilesMatch(expected, missingCache, build), false);
    assert.equal(windowsLlvmPrehookConfiguredFilesMatch(expected, expected, build), true);
  });

  test("pre-hook output inventory uses the proved checkpoint names and always rejects PE names", () => {
    const oldIndex = ".cmake/api/v1/reply/index-2026-11-04T19-55-22-0131.json",
      newIndex = ".cmake/api/v1/reply/index-2026-11-05T23-37-49-0006.json",
      checkpoint = {[oldIndex]: "mode:644:sha256:" + "a".repeat(64), "lib/fixture.obj": "mode:644:sha256:" + "b".repeat(64)},
      savedNames = Object.keys(checkpoint).sort(), actualNames = [newIndex, "lib/fixture.obj"].sort(),
      savedInventory = {count: savedNames.length, sha256: createHash("sha256").update(JSON.stringify(savedNames)).digest("hex")};
    assert.deepEqual(windowsLlvmAdmissionOutputInventory(actualNames, checkpoint, true), savedInventory);
    assert.notDeepEqual(windowsLlvmAdmissionOutputInventory(actualNames, checkpoint, false), savedInventory,
      "ordinary admission still uses actual live names");
    assert.throws(() => windowsLlvmAdmissionOutputInventory([...actualNames, "CMakeFiles/CompilerId.exe"], checkpoint, true), /PE tool or DLL/);
  });

  test("pre-hook refresh snapshots only the admitted cache metadata for the configure race guard", t => {
    const f = fixture(t), plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output,
        {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, 1), build = plan.build,
      compilerFiles = path.join(build, "CMakeFiles", "3.29.6"), cache = path.join(build, "CMakeCache.txt"),
      buildNinja = path.join(build, "build.ninja"), rules = path.join(build, "CMakeFiles", "rules.ninja"),
      database = path.join(build, "compile_commands.json"), cCompiler = path.join(compilerFiles, "CMakeCCompiler.cmake"),
      cxxCompiler = path.join(compilerFiles, "CMakeCXXCompiler.cmake");
    fs.mkdirSync(compilerFiles, {recursive: true});
    fs.writeFileSync(cache, "pre-hook cache\n"); fs.writeFileSync(buildNinja, "stable build graph\n");
    fs.writeFileSync(rules, "stable rules\n"); writeJson(database, []);
    fs.writeFileSync(cCompiler, 'set(CMAKE_C_COMPILER_ID "IntelLLVM")\n');
    fs.writeFileSync(cxxCompiler, 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")\n');
    const saved = buildFileInputs([cache, buildNinja, rules, database, cCompiler, cxxCompiler]);
    assert.deepEqual(windowsLlvmRefreshConfiguredFiles(saved, plan, false), saved, "same-hook refresh keeps its exact prior hashes");
    fs.writeFileSync(cache, "pre-hook cache with harmless metadata refresh\n");
    const current = windowsLlvmRefreshConfiguredFiles(saved, plan, true);
    assert.notEqual(current[cache], saved[cache]);
    assert.equal(windowsLlvmPrehookConfiguredFilesMatch(saved, current, build), true);
    fs.writeFileSync(buildNinja, "unexpected graph drift\n");
    assert.throws(() => windowsLlvmRefreshConfiguredFiles(saved, plan, true), /changed after admission/);
  });

  test("uses the size-optimized CMake configuration for compiler host libraries", t => {
    const f = fixture(t), plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output,
      {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, 4);
    const buildTypes = plan.configure.filter(argument => argument.startsWith("-DCMAKE_BUILD_TYPE="));
    assert.deepEqual(buildTypes, ["-DCMAKE_BUILD_TYPE=MinSizeRel"]);
  });

  test("pins a static Intel/COFF plan and brackets hostile-looking literal paths safely", t => {
    const f = fixture(t), plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output,
      {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, 4);
    assert.equal(plan.archiveSha256, archiveSha256);
    assert.equal(plan.targetTriple, "x86_64-pc-windows-msvc");
    assert.equal(plan.compilerTools.CMAKE_C_COMPILER, f.descriptor.tools.clangCl);
    assert.equal(plan.compilerTools.CMAKE_CXX_COMPILER, f.descriptor.tools.clangCl);
    assert.equal(plan.compilerTools.CMAKE_LINKER, f.descriptor.tools.lldLink, "linking stays on the explicit Intel rule override and selected lld-link");
    assert.equal(Object.hasOwn(plan.nativeTools, "llvm-mt"), false, "the static plan does not require the libxml2-gated llvm-mt tool");
    assert.equal(Object.hasOwn(plan.compilerTools, "CMAKE_MT"), false, "manifest tool is not part of the static linker contract");
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_LIBXML2=OFF"));
    assert.doesNotMatch(plan.toolchain, /CMAKE_MT|llvm-mt/);
    assert.ok(plan.toolchain.includes("[==[" + f.descriptor.tools.clangCl + "]==]"));
    const flags = plan.toolchain.split("\n").find(line => line.startsWith("set(CMAKE_C_FLAGS_INIT "));
    assert.ok(flags?.startsWith("set(CMAKE_C_FLAGS_INIT [===["), "the CMake delimiter must avoid the path's internal ]==]");
    assert.ok(flags?.endsWith("]===])"));
    assert.deepEqual(plan.phases, buildPhases);
    assert.ok(plan.configure.includes("-DLLVM_TARGETS_TO_BUILD=AMDGPU;NVPTX;X86"));
    assert.ok(plan.configure.includes("-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + plan.sourceOptionsFile));
    assert.equal(plan.sourceOptionsFile, path.join(f.output, "llvm-host-source-options.cmake"));
    assert.match(plan.sourceOptionsText, /cmake_language\(DEFER CALL _mom_llvm_amdgpulibcalls_host_option\)/);
    assert.match(plan.sourceOptionsText, /APPEND PROPERTY COMPILE_OPTIONS \[\[-fno-builtin\]\]/);
    assert.ok(plan.sourceOptionsText.includes("[[" + path.join(plan.source, "llvm/lib/Target/AMDGPU/AMDGPULibCalls.cpp") + "]]"));
    assert.doesNotMatch(plan.configure.join(" "), /LLD_INCLUDE_TESTS|CLANG_ENABLE_REPL|LLVM_ENABLE_TERMINFO/);
    assert.ok(plan.toolchain.includes("/MT"));
    assert.ok(plan.toolchain.includes("/EHsc"));
    assert.ok(plan.toolchain.includes("/GR"));
    assert.ok(plan.toolchain.includes("CMAKE_INCLUDE_SYSTEM_FLAG_C [==[/external:I]==]"));
    assert.ok(plan.toolchain.includes("CMAKE_INCLUDE_SYSTEM_FLAG_CXX [==[/external:I]==]"));
    assert.ok(plan.ruleOverrideText.includes("set(CMAKE_INCLUDE_SYSTEM_FLAG_C /external:I)"));
    assert.ok(plan.ruleOverrideText.includes("set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX /external:I)"));
    assert.ok(plan.toolchain.includes("set(CMAKE_ASM_MASM_FLAGS [==[/nologo --m64]==] CACHE STRING [==[]==] FORCE)"));
  });

  test("requires exactly one AMDGPULibCalls-only host compile option", t => {
    const f = fixture(t), source = path.join(f.root, "upstream"), build = path.join(f.root, "build"),
      target = path.join(source, "llvm/lib/Target/AMDGPU/AMDGPULibCalls.cpp"),
      other = path.join(source, "llvm/lib/Target/AMDGPU/AMDGPULibCallsTest.cpp");
    fs.mkdirSync(path.dirname(target), {recursive: true}); fs.mkdirSync(build);
    fs.writeFileSync(target, "int target();\n"); fs.writeFileSync(other, "int other();\n");
    const database = path.join(build, "compile_commands.json"), rows = [
      {directory: build, file: target, arguments: ["icx", target]},
      {directory: build, file: other, arguments: ["icx", other]},
    ];
    writeJson(database, rows);
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(build, source), /unexpected host-only -fno-builtin state/,
      "the prior unscoped compile database must fail before the source hook");
    rows[0]?.arguments?.push("-fno-builtin"); writeJson(database, rows);
    assert.deepEqual(verifyAmdgpulibCallsCompileFlag(build, source),
      {path: target, flag: "-fno-builtin", compileCommands: 2});
    rows[1]?.arguments?.push("-fno-builtin"); writeJson(database, rows);
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(build, source), /escaped the AMDGPULibCalls.cpp source row/,
      "the option must not leak to another AMDGPU source");
    const first = rows[0]; assert.ok(first);
    rows.pop(); rows.push({...first}); writeJson(database, rows);
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(build, source), /exactly one AMDGPULibCalls.cpp/,
      "duplicate compile rows for the selected source must not pass");
  });

  test("CMake parses generated bracket arguments without interpreting their contents", t => {
    const f = fixture(t), plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output,
      {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, 4);
    const probe = spawnSync("cmake", ["--version"], {encoding: "utf8"});
    if (probe.error || probe.status !== 0) {return t.skip("CMake unavailable for literal toolchain syntax check");}
    const script = path.join(f.root, "parse-toolchain.cmake");
    fs.writeFileSync(script, plan.toolchain);
    const result = spawnSync("cmake", ["-P", script], {encoding: "utf8"});
    assert.equal(result.status, 0, (result.stdout || "") + (result.stderr || ""));
  });

  test("Windows feature probes link against the selected SDK instead of accepting unresolved declarations", t => {
    const f = fixture(t), plan = windowsLlvmSdkPlan(f.descriptor, f.archive, f.output,
      {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}, 2);
    assert.match(plan.toolchain, /set\(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE\)/);
    assert.doesNotMatch(plan.toolchain, /CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY/);
    for (const name of ["C", "CXX"]) {
      assert.ok(plan.toolchain.includes("set(CMAKE_" + name + "_COMPILER_TARGET [==[x86_64-pc-windows-msvc]==]"));
      assert.ok(plan.ruleOverrideText.includes("set(CMAKE_" + name + "_STANDARD_LIBRARIES_INIT "));
    }
    assert.ok(plan.ruleOverrideText.includes(f.descriptor.windowsSdk.importLibraries[0] || "missing"));
  });

  test("hashes only the upstream source roots and actual top-level LLVM license", t => {
    const f = fixture(t), source = path.join(f.root, "upstream");
    for (const name of ["llvm", "clang", "lld", "cmake", "third-party", "libunwind/include"]) {
      const directory = path.join(source, name); fs.mkdirSync(directory, {recursive: true});
      fs.writeFileSync(path.join(directory, "input.txt"), name + " source\n");
    }
    const compactUnwind = path.join(source, "libunwind", "include", "mach-o", "compact_unwind_encoding.h");
    fs.mkdirSync(path.dirname(compactUnwind), {recursive: true}); fs.writeFileSync(compactUnwind, "pinned libunwind header\n");
    fs.writeFileSync(path.join(source, "LICENSE.TXT"), "upstream license\n");
    assert.equal(fs.existsSync(path.join(source, "CMakeLists.txt")), false);
    const before = llvmSourceSnapshot(source);
    fs.writeFileSync(compactUnwind, "changed pinned libunwind header\n");
    assert.notEqual(llvmSourceSnapshot(source).sha256, before.sha256, "libunwind headers must be source-bound");
  });

  test("requires the libunwind include subtree in the archive catalog", t => {
    const f = fixture(t), upstream = path.join(f.root, "llvm-project-21.1.8.src");
    for (const name of ["llvm", "clang", "lld", "cmake", "third-party", "libunwind/include"]) {
      fs.mkdirSync(path.join(upstream, name), {recursive: true});
    }
    fs.writeFileSync(path.join(upstream, "LICENSE.TXT"), "upstream license\n");
    const archive = path.join(f.root, "catalog.tar");
    /** @param {boolean} includeUnwind */
    const createArchive = includeUnwind => {
      if (!includeUnwind) {fs.rmSync(path.join(upstream, "libunwind", "include"), {recursive: true});}
      const result = spawnSync("/usr/bin/tar", ["-cf", archive, "--directory", f.root, "llvm-project-21.1.8.src"], {encoding: "utf8"});
      assert.equal(result.status, 0, result.stderr || result.error?.message);
    };
    createArchive(true);
    assert.ok(verifyArchiveCatalog(archive).members > 0);
    createArchive(false);
    assert.throws(() => verifyArchiveCatalog(archive), /omitted required source: libunwind\/include/);
  });

  test("resolves File API target paths relative to absolute model roots", t => {
    const f = fixture(t), source = path.join(f.root, "source"), llvm = path.join(source, "llvm"), build = path.join(f.root, "build");
    for (const project of ["llvm", "clang", "lld"]) {fs.mkdirSync(path.join(source, project), {recursive: true});}
    const reply = path.join(build, ".cmake", "api", "v1", "reply");
    fs.mkdirSync(reply, {recursive: true});
    /** @type {[string, string, string, string, string][]} */
    const targets = [
      ["LLVMAMDGPUCodeGen", "llvm", "lib/Target/AMDGPU", "lib/LLVMAMDGPUCodeGen.lib", "STATIC_LIBRARY"],
      ["LLVMNVPTXCodeGen", "llvm", "lib/Target/NVPTX", "lib/LLVMNVPTXCodeGen.lib", "STATIC_LIBRARY"],
      ["LLVMX86CodeGen", "llvm", "lib/Target/X86", "lib/LLVMX86CodeGen.lib", "STATIC_LIBRARY"],
      ["clangSema", "clang", "../clang/lib/Sema", "lib/clangSema.lib", "STATIC_LIBRARY"],
      ["lldELF", "lld", "../lld/lib/ELF", "lib/lldELF.lib", "STATIC_LIBRARY"],
      // Configured-but-unselected executables are ignored; build output audit separately
      // rejects one if it enters a requested aggregate.
      ["clang", "clang", "../clang/tools/driver", "bin/clang.exe", "EXECUTABLE"],
    ];
    const references = targets.map(([name, , sourcePath, artifact, type], index) => {
      const jsonFile = "target-" + index + ".json";
      writeJson(path.join(reply, jsonFile), {name, type, paths: {source: sourcePath, build: "target-" + index},
        artifacts: [{path: artifact}], sources: []});
      return {id: name + ":" + type, jsonFile};
    });
    writeJson(path.join(reply, "index-test.json"), {reply: {"codemodel-v2": {jsonFile: "codemodel.json"}}});
    writeJson(path.join(reply, "codemodel.json"), {paths: {source: llvm, build}, configurations: [{targets: references}]});
    const rows = llvmStaticArchiveTargets(build, source);
    assert.equal(rows.length, 5);
    assert.deepEqual(new Set(rows.map(row => row.project)), new Set(["llvm", "clang", "lld"]));
    assert.equal(rows.find(row => row.name === "clangSema")?.sourceDirectory, path.join(source, "clang", "lib", "Sema"));
    assert.equal(rows.some(row => row.name === "clang"), false);
  });

  test("keeps backend phases and the complete core-library aggregate as bounded named phases", t => {
    const f = fixture(t), build = path.join(f.root, "build"); fs.mkdirSync(build);
    fs.writeFileSync(path.join(build, "build.ninja"), buildPhases.map(row => "build " + row.target + ": phony\n").join(""));
    assert.equal(verifyBuildPhases(build).phases.length, 6);
    assert.deepEqual(buildPhases.map(row => row.name), ["llvm-amdgpu", "llvm-nvptx", "llvm-x86", "llvm-core", "clang", "lld"]);
    assert.equal(buildPhases.find(row => row.name === "llvm-core")?.target, "llvm-libraries");
    assert.equal(buildPhases.every(row => row.target.length > 0), true);
    fs.writeFileSync(path.join(build, "build.ninja"), buildPhases.map(row =>
      "build " + (row.name === "llvm-core" ? "lib/all" : row.target) + ": phony\n").join(""));
    assert.throws(() => verifyBuildPhases(build), /CMake omitted LLVM 21 phase target: llvm-libraries/);
  });

  test("requires IntelLLVM compiler identity files for both languages", t => {
    const f = fixture(t), build = path.join(f.root, "build"), cmakeFiles = path.join(build, "CMakeFiles", "3.29.6");
    fs.mkdirSync(cmakeFiles, {recursive: true});
    fs.writeFileSync(path.join(cmakeFiles, "CMakeCCompiler.cmake"), 'set(CMAKE_C_COMPILER_ID "IntelLLVM")\n');
    fs.writeFileSync(path.join(cmakeFiles, "CMakeCXXCompiler.cmake"), 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")\n');
    assert.equal(Object.values(verifyIntelCompilerIdentity(build)).length, 2);
    fs.writeFileSync(path.join(cmakeFiles, "CMakeCXXCompiler.cmake"), 'set(CMAKE_CXX_COMPILER_ID "Clang")\n');
    assert.throws(() => verifyIntelCompilerIdentity(build), /selected IntelLLVM driver/);
  });

  test("audits Intel C++ and llvm-ml x64 commands against real COFF object headers", t => {
    const f = fixture(t), source = path.join(f.root, "upstream"), build = path.join(f.root, "build"),
      cpp = path.join(source, "llvm/lib/Support/Hash.cpp"), asm = path.join(source, "llvm/lib/Support/Hash-x86_64.asm"),
      cppObject = path.join(build, "hash.obj"), asmObject = path.join(build, "hash-x86_64.obj"),
      compiler = path.join(f.root, "intel compiler", "icx-cl"), masm = path.join(f.root, "llvm", "llvm-ml");
    fs.mkdirSync(path.dirname(cpp), {recursive: true}); fs.mkdirSync(build, {recursive: true});
    fs.writeFileSync(cpp, "int hash();\n"); fs.writeFileSync(asm, "PUBLIC hash\n");
    const object = Buffer.alloc(20); object.writeUInt16LE(0x8664, 0); fs.writeFileSync(cppObject, object); fs.writeFileSync(asmObject, object);
    /** @param {string} value */ const quote = value => /\s/.test(value) ? '"' + value + '"' : value;
    const entries = [
      {directory: build, file: cpp, output: path.basename(cppObject), command: [compiler, "--target=x86_64-pc-windows-msvc", "/MT",
        "/external:I" + path.join(f.root, "windows sdk", "include"), "/EHsc", "/GR", "/c", cpp, "-o", cppObject].map(quote).join(" ")},
      {directory: build, file: asm, output: path.basename(asmObject), command: [masm, "--m64", "-c", asm, "-Fo", asmObject].map(quote).join(" ")},
    ];
    const database = path.join(build, "compile_commands.json"); fs.writeFileSync(database, JSON.stringify(entries));
    const audit = auditCompileCommands(database, compiler, build, source, [cppObject, asmObject]);
    assert.deepEqual(audit.missing, []);
    assert.equal(audit.commands[cppObject]?.language, "CXX");
    assert.equal(audit.commands[asmObject]?.language, "ASM_MASM");
    assert.equal(audit.objects.length, 1);
    assert.equal(audit.objects[0]?.compilerIdentity, "IntelLLVM");
    assert.match(audit.objects[0]?.abiSha256 || "", /^[0-9a-f]{64}$/);
  });

  test("keeps the configured receipt immutable and audits only a complete six-phase checkpoint", t => {
    const f = fixture(t), output = f.output, cmake = path.join(f.root, "fake-cmake"), ninja = path.join(f.root, "fake-ninja"),
      descriptorFile = path.join(f.root, "descriptor.json"), archive = path.join(f.root, "llvm-project-21.1.8.src.tar.xz");
    /** @param {string} file @param {string} [value] */ const writeFile = (file, value = "fixture\n") => {
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value);
    };
    writeFile(cmake, '#!/bin/sh\nmkdir -p "$2/lib"\nfor name in LLVMAMDGPUCodeGen LLVMNVPTXCodeGen LLVMX86CodeGen clangSema lldELF; do printf "archive\\n" > "$2/lib/$name.lib"; done\nprintf "%s\\n" "$4" > "$2/phase.stamp"\n'); fs.chmodSync(cmake, 0o700);
    writeFile(ninja); writeFile(archive); writeJson(descriptorFile, f.descriptor);
    for (const file of [f.descriptor.tools.clangCl, f.descriptor.tools.lldLink, f.descriptor.tools.llvmNm,
      f.descriptor.tools.llvmReadobj, ...["llvm-lib", "llvm-ranlib", "llvm-rc", "llvm-ml", "llvm-tblgen", "clang-tblgen"]
        .map(name => path.join(path.dirname(f.descriptor.tools.llvmNm), name)),
      ...f.descriptor.windowsSdk.msvcLibraries, ...f.descriptor.windowsSdk.ucrtLibraries, ...f.descriptor.windowsSdk.importLibraries]) {writeFile(file);}
    fs.mkdirSync(f.descriptor.tools.clangResourceInclude, {recursive: true});
    writeFile(path.join(f.descriptor.tools.clangResourceInclude, "stddef.h"));
    writeFile(path.join(f.root, "sdk]==]bracket", "include", "windows.h"));

    const tools = {cmake, ninja}, plan = windowsLlvmSdkPlan(f.descriptor, archive, output, tools, 1),
      savedPlan = {descriptorFile, archive, source: plan.source, output, tools, jobs: 1,
        sourceOptionsFile: plan.sourceOptionsFile, sourceOptionsText: plan.sourceOptionsText};
    fs.mkdirSync(output, {mode: 0o700});
    fs.mkdirSync(plan.source, {mode: 0o700});
    for (const name of ["llvm", "clang", "lld", "cmake", "third-party", "libunwind/include"]) {fs.mkdirSync(path.join(plan.source, name), {recursive: true});}
    writeFile(path.join(plan.source, "LICENSE.TXT"));
    const amdgpulibCalls = path.join(plan.source, "llvm/lib/Target/AMDGPU/AMDGPULibCalls.cpp"),
      amdgpulibCallsOther = path.join(plan.source, "llvm/lib/Target/AMDGPU/AMDGPULibCallsOther.cpp");
    writeFile(amdgpulibCalls, "int calls();\n"); writeFile(amdgpulibCallsOther, "int other();\n");
    const build = plan.build, reply = path.join(build, ".cmake", "api", "v1", "reply"), compilerFiles = path.join(build, "CMakeFiles", "3.29.6");
    fs.mkdirSync(build, {mode: 0o700});
    fs.mkdirSync(reply, {recursive: true}); fs.mkdirSync(compilerFiles, {recursive: true});
    writeFile(path.join(build, "build.ninja")); writeFile(path.join(build, "CMakeFiles", "rules.ninja"));
    writeFile(path.join(compilerFiles, "CMakeCCompiler.cmake"), 'set(CMAKE_C_COMPILER_ID "IntelLLVM")\n');
    writeFile(path.join(compilerFiles, "CMakeCXXCompiler.cmake"), 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")\n');
    writeJson(path.join(build, "compile_commands.json"), [
      {directory: build, file: amdgpulibCalls, arguments: [f.descriptor.tools.clangCl, "-fno-builtin", amdgpulibCalls]},
      {directory: build, file: amdgpulibCallsOther, arguments: [f.descriptor.tools.clangCl, amdgpulibCallsOther]},
    ]);
    const cache = {CMAKE_C_COMPILER: plan.compilerTools.CMAKE_C_COMPILER, CMAKE_CXX_COMPILER: plan.compilerTools.CMAKE_CXX_COMPILER,
      CMAKE_LINKER: plan.compilerTools.CMAKE_LINKER, CMAKE_AR: plan.compilerTools.CMAKE_AR, CMAKE_RANLIB: plan.compilerTools.CMAKE_RANLIB,
      CMAKE_MAKE_PROGRAM: ninja, CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded", CMAKE_ASM_MASM_COMPILER: plan.nativeTools["llvm-ml"],
      CMAKE_BUILD_TYPE: "MinSizeRel",
      CMAKE_PROJECT_TOP_LEVEL_INCLUDES: plan.sourceOptionsFile,
      CMAKE_C_COMPILER_TARGET: "x86_64-pc-windows-msvc", CMAKE_CXX_COMPILER_TARGET: "x86_64-pc-windows-msvc",
      CMAKE_ASM_MASM_FLAGS: "/nologo --m64", CMAKE_INCLUDE_SYSTEM_FLAG_C: "/external:I", CMAKE_INCLUDE_SYSTEM_FLAG_CXX: "/external:I",
      LLVM_ENABLE_PROJECTS: "clang;lld", LLVM_TARGETS_TO_BUILD: "AMDGPU;NVPTX;X86", LLVM_ENABLE_EH: "ON", LLVM_ENABLE_RTTI: "ON",
      LLVM_ENABLE_PLUGINS: "OFF", LLVM_BUILD_TOOLS: "OFF", CLANG_BUILD_TOOLS: "OFF", LLD_BUILD_TOOLS: "OFF", BUILD_SHARED_LIBS: "OFF",
      LLVM_BUILD_LLVM_DYLIB: "OFF", LLVM_BUILD_LLVM_C_DYLIB: "OFF", LLVM_LINK_LLVM_DYLIB: "OFF", LIBCLANG_BUILD_STATIC: "ON"};
    const cacheFile = path.join(build, "CMakeCache.txt"), cacheText = Object.entries(cache).map(([key, value]) => `${key}:STRING=${value}\n`).join("");
    writeFile(cacheFile, cacheText.replace(/^CMAKE_ASM_MASM_FLAGS:STRING=.*\n/m, ""));
    assert.throws(() => verifyCMakeCache(build, plan), /CMAKE_ASM_MASM_FLAGS/,
      "the strict cache contract rejects an absent MASM language flag");
    writeFile(cacheFile, cacheText);
    assert.equal(verifyCMakeCache(build, plan), cacheFile);
    const targetData = [
      ["LLVMAMDGPUCodeGen", "llvm/lib/Target/AMDGPU", "lib/LLVMAMDGPUCodeGen.lib"],
      ["LLVMNVPTXCodeGen", "llvm/lib/Target/NVPTX", "lib/LLVMNVPTXCodeGen.lib"],
      ["LLVMX86CodeGen", "llvm/lib/Target/X86", "lib/LLVMX86CodeGen.lib"],
      ["clangSema", "clang/lib/Sema", "lib/clangSema.lib"], ["lldELF", "lld/lib/ELF", "lib/lldELF.lib"],
    ];
    const references = targetData.map(([name, source, artifact], index) => {
      const jsonFile = `target-${index}.json`;
      writeJson(path.join(reply, jsonFile), {name, type: "STATIC_LIBRARY", paths: {source, build: `target-${index}`},
        artifacts: [{path: artifact}], sources: []});
      return {id: `${name}:STATIC_LIBRARY`, jsonFile};
    });
    writeJson(path.join(reply, "index-test.json"), {reply: {"codemodel-v2": {jsonFile: "codemodel.json"}}});
    writeJson(path.join(reply, "codemodel.json"), {paths: {source: plan.source, build}, configurations: [{targets: references}]});
    const targets = llvmStaticArchiveTargets(build, plan.source), sourceState = llvmSourceSnapshot(plan.source),
      catalog = {members: 1, sha256: createHash("sha256").update("fixture archive catalog").digest("hex")};
    const configuredInputFiles = windowsLlvmConfiguredInputFiles(descriptorFile, archive, plan),
      planFile = configuredInputFiles.generated.plan;
    writeJson(planFile, savedPlan); fs.copyFileSync(archive, configuredInputFiles.generated.archiveCopy);
    writeFile(configuredInputFiles.generated.overlay, plan.overlayText);
    writeFile(configuredInputFiles.generated.ruleOverride, plan.ruleOverrideText);
    writeFile(configuredInputFiles.generated.toolchainFile, plan.toolchain);
    writeFile(configuredInputFiles.generated.sourceOptionsFile, plan.sourceOptionsText);
    const fingerprints = buildFileInputs(configuredInputFiles.base);
    for (const file of Object.values(configuredInputFiles.generated)) {fingerprints[file] = fileHash(file);}
    const producerSnapshotFile = path.join(output, "producer-snapshot.js");
    fs.copyFileSync(path.join(__dirname, "../scripts/build-native-windows-llvm21-sdk.js"), producerSnapshotFile);
    const producerSnapshot = {path: producerSnapshotFile, sha256: fileHash(producerSnapshotFile)};
    const configuredFiles = windowsLlvmConfiguredFiles(plan);
    const recipeSha256 = createHash("sha256").update(JSON.stringify({configure: plan.configure, toolchain: plan.toolchain,
      rules: plan.ruleOverrideText, overlay: plan.overlayText, sourceOptionsFile: plan.sourceOptionsFile,
      sourceOptionsText: plan.sourceOptionsText, sourceState, catalog})).digest("hex");
    const sourceOptions = {path: plan.sourceOptionsFile, sha256: fileHash(plan.sourceOptionsFile)},
      hostSourceFlag = {path: amdgpulibCalls, flag: "-fno-builtin", compileCommands: 2};
    const receipt = {version: 1, role: plan.role, runId: "fixture-build-run", stage: "configured", plan: savedPlan,
      planSha256: fileHash(planFile), sourceState, catalog, fingerprints, recipeSha256, configure: plan.configure,
      producerSnapshot, sourceOptions, hostSourceFlag, buildOnly: true, compiled: false, executed: false, publicationQualified: false, targets, phases: buildPhases,
      outputs: jitBuildOutputs(build), configuredFiles, compileCommandsSha256: fileHash(path.join(build, "compile_commands.json"))};
    const receiptFile = path.join(output, "result.json"); writeJson(receiptFile, receipt);
    const configuredReceiptBytes = fs.readFileSync(receiptFile, "utf8");

    for (const phase of buildPhases) {buildWindowsLlvmSdk(output, phase.name, {MOM_AOT_BUILD_ONLY: "1", PATH: process.env["PATH"]});}
    assert.equal(fs.readFileSync(receiptFile, "utf8"), configuredReceiptBytes);
    const state = JSON.parse(fs.readFileSync(path.join(output, "build-state.json"), "utf8"));
    const currentFile = path.join(output, "result-current.json"), current = JSON.parse(fs.readFileSync(currentFile, "utf8"));
    assert.equal(state.stage, "built"); assert.equal(state.completedPhases.length, 6);
    assert.equal(current.role, "windows-llvm21-current-result"); assert.equal(current.phaseProofs.length, 6);
    assert.deepEqual(current.sourceOptions, sourceOptions); assert.deepEqual(current.hostSourceFlag, hostSourceFlag);
    assert.deepEqual(current.originalReceipt, {path: receiptFile, sha256: fileHash(receiptFile)});
    assert.equal(current.buildOnly, true); assert.equal(current.executed, false); assert.equal(current.publicationQualified, false);
    assert.notDeepEqual(receipt.outputs, state.outputs, "compilation adds files after the configure snapshot");
    assert.throws(() => verifyWindowsLlvmAdmission(output, {MOM_AOT_BUILD_ONLY: "1", PATH: process.env["PATH"]}),
      /configured static target graph changed/, "admission checks completed bytes before rejecting this intentionally incomplete five-target SDK");

    const currentBytes = fs.readFileSync(currentFile, "utf8");
    writeJson(currentFile, {...current, inputs: {...current.inputs, [path.join(output, "unselected-input")]: "a".repeat(64)}});
    assert.throws(() => auditWindowsLlvmAdmission(output, {MOM_AOT_BUILD_ONLY: "1", PATH: process.env["PATH"]}), /selected source or tool input changed/);
    fs.writeFileSync(currentFile, currentBytes);
    fs.rmSync(currentFile);
    assert.throws(() => auditWindowsLlvmAdmission(output, {MOM_AOT_BUILD_ONLY: "1", PATH: process.env["PATH"]}), /configured static target graph changed/);
    assert.equal(fs.existsSync(currentFile), true, "audit reconstructs the current record from the complete phase checkpoint");
    assert.equal(fs.existsSync(path.join(output, "admitted.json")), false, "fake CMake output is not admitted as an SDK");
    assert.equal(fs.readFileSync(receiptFile, "utf8"), configuredReceiptBytes);
  });

  test("refuses to invent an admission when current build evidence is absent", t => {
    const f = fixture(t); fs.mkdirSync(f.output, {mode: 0o700});
    writeJson(path.join(f.output, "result.json"), {version: 1, stage: "configured", buildOnly: true, executed: false,
      publicationQualified: false});
    assert.throws(() => verifyWindowsLlvmAdmission(f.output, {MOM_AOT_BUILD_ONLY: "1"}), /configured receipt is missing its plan/);
    assert.equal(fs.existsSync(path.join(f.output, "admitted.json")), false);
    assert.equal(fs.existsSync(path.join(f.output, "admitted-objects.json")), false);
    const refreshed = path.join(f.root, "should-not-exist");
    assert.throws(() => refreshWindowsLlvmSdk(f.output, refreshed, {MOM_AOT_BUILD_ONLY: "1"}), /completed admitted output/);
    assert.equal(fs.existsSync(refreshed), false, "failed refresh preflight must not create its destination");
  });
});
