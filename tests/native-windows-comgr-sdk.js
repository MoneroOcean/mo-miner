"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {spawnSync} = require("node:child_process");
const {revision, hostSources, recipeFiles, windowsComgrSdkPlan, verifyWindowsDeviceResources, verifyIntelCompilerIdentity,
  verifyNativeIntelCompilerIdentity, verifyNativeClangIdentity, verifyNativeLibrarianIdentity, auditBc2hCompileCommands, verifyComgrCache, auditComgrCompileCommands, comgrArchive,
  auditComgrArchiveMembers, nativeRuntimeInputs, cleanEnvironment, canonicalNativeClang} = require("../scripts/build-native-windows-comgr-sdk");
const {fileHash, buildFileInputs, sourceInputs} = require("../scripts/build-source-inputs");
const {verifyWindowsLlvmAdmission} = require("../scripts/build-native-windows-llvm-sdk");
const {tempDir} = require("./common/temp-dir");

const repo = path.resolve(__dirname, "..");

/** @param {string} name */
function cmakeVar(name) {return ["$", "{", name, "}"].join("");}

/** Small explicit inputs; no toolchain, VM, network or GPU is invoked. */
/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = tempDir(t, "mom-windows-comgr-");
  const include = path.join(root, "sdk]==]bracket", "include"), library = path.join(root, "windows sdk", "lib"),
    resource = path.join(root, "Intel 2026.2", "clang-resource", "include");
  fs.mkdirSync(include, {recursive: true}); fs.mkdirSync(library, {recursive: true}); fs.mkdirSync(resource, {recursive: true});
  const mapped = path.join(include, "windows.h"); fs.writeFileSync(mapped, "fixture header\n");
  const overlay = path.join(root, "windows sdk overlay.json");
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, roots: [
    {type: "file", name: mapped, "external-contents": mapped},
  ]}));
  const descriptor = {version: 1,
    tools: {clangCl: path.join(root, "Intel 2026.2", "bin", "icx-cl"), lldLink: path.join(root, "LLVM", "bin", "lld-link"),
      shermaot: path.join(root, "shermes"), llvmNm: path.join(root, "LLVM", "bin", "llvm-nm"),
      llvmReadobj: path.join(root, "LLVM", "bin", "llvm-readobj"), clangResourceInclude: resource},
    hermes: {source: path.join(root, "hermes source"), build: path.join(root, "hermes build"), shermaot: path.join(root, "shermes")},
    windowsSdk: {overlay, sourceRoot: path.join(root, "windows sdk"), includeRoots: [include], libraryRoots: [library],
      msvcLibraries: [path.join(library, "libcmt.lib")], ucrtLibraries: [path.join(library, "libucrt.lib")],
      importLibraries: [path.join(library, "kernel32.lib")], allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: path.join(root, "source include"), ssl: path.join(library, "ssl.lib"), crypto: path.join(library, "crypto.lib"),
      libuv: path.join(library, "uv.lib"), zlib: path.join(library, "zlib.lib")}};
  const descriptorFile = path.join(root, "windows-descriptor.json"); fs.writeFileSync(descriptorFile, JSON.stringify(descriptor));
  const input = {source: path.join(root, "rocm llvm git"), descriptorFile, descriptor, targetLlvm: path.join(root, "windows llvm output"),
    nativeLlvm: path.join(root, "native llvm 21"), hostCompiler: path.join(root, "Intel 2026.2", "bin", "icx"),
    deviceReceipt: path.join(root, "windows resources", "result.json"), deviceInstall: path.join(root, "windows resources", "install"),
    output: path.join(root, "new comgr output"), jobs: 2, tools: {cmake: "/tools/cmake", ninja: "/tools/ninja"}};
  return {root, input, include, descriptor, overlay};
}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

/** @param {import("node:test").TestContext} t */
function deviceReceiptFixture(t) {
  const root = tempDir(t, "mom-windows-comgr-device-");
  const sourceRoot = path.join(root, "source"), sourceFile = path.join(sourceRoot, "device", "source.cl"), hiprtcSource = path.join(sourceRoot, "hiprtc", "source.cpp"),
    install = path.join(root, "install"), bitcode = path.join(install, "amdgcn", "bitcode"), cmake = path.join(install, "lib", "cmake", "AMDDeviceLibs");
  fs.mkdirSync(path.dirname(sourceFile), {recursive: true}); fs.mkdirSync(path.dirname(hiprtcSource), {recursive: true});
  fs.mkdirSync(bitcode, {recursive: true}); fs.mkdirSync(cmake, {recursive: true});
  fs.writeFileSync(sourceFile, "device source\n"); fs.writeFileSync(hiprtcSource, "HIPRTC source\n");
  const names = ["asanrtl.bc", "hip.bc", "ockl.bc", "ocml.bc", "opencl.bc", "oclc_abi_version_400.bc",
    ...Array.from({length: 65}, (_, i) => "oclc_fixture_" + i + ".bc")].sort();
  for (const name of names) {fs.writeFileSync(path.join(bitcode, name), "verified bitcode: " + name + "\n");}
  const configFile = path.join(cmake, "AMDDeviceLibsConfig.cmake");
  fs.writeFileSync(configFile, names.map(name => `add_library(${name.slice(0, -3)} STATIC IMPORTED)\nset_target_properties(${name.slice(0, -3)} PROPERTIES\n  IMPORTED_LOCATION "\${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode/${name}")\n`).join(""));
  const recipe = {files: {}, graph: {"scripts/retained-old-recipe.js": "a".repeat(64)}},
    inputsFile = path.join(root, "input.bin"), dependencyFile = path.join(root, "dependency.bin");
  fs.writeFileSync(inputsFile, "input\n"); fs.writeFileSync(dependencyFile, "dependency\n");
  const header = path.join(root, "hipRTC"), object = path.join(root, "hipRTC_header.obj");
  fs.writeFileSync(header, "header\n"); fs.writeFileSync(object, "object\n");
  const device = buildFileInputs([...names.map(name => path.join(bitcode, name)), configFile]);
  const receipt = {version: 1, target: "win32", revision, clrRevision: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50", recipe,
    source: sourceInputs(sourceRoot, ["device", "hiprtc"], []).files, inputs: buildFileInputs([inputsFile]), dependencies: buildFileInputs([dependencyFile]),
    device, header: {path: header, sha256: fileHash(header)}, object: {path: object, sha256: fileHash(object)},
    buildOnly: true, executed: false, publicationQualified: false};
  const receiptFile = path.join(root, "result.json"); writeJson(receiptFile, receipt);
  return {root, receiptFile, install, bitcode, names, configFile};
}

/** @param {import("node:test").TestContext} t */
function pchPatchFixture(t) {
  const root = tempDir(t, "mom-comgr-pch-patch-");
  const source = path.join(root, "source"), cmakeDir = path.join(source, "amd/comgr/cmake"), pchRule = path.join(cmakeDir, "opencl_pch.cmake"),
    patchFile = path.join(root, "cross-tools-pch.patch"), project = path.join(root, "project"),
    targetClang = path.join(root, "Windows LLVM 21/bin/clang.exe"), nativeClang = path.join(root, "native LLVM 21/bin/clang"),
    resourceDir = path.join(root, "native LLVM 21/lib/clang/21"), header = path.join(resourceDir, "include/opencl-c.h");
  fs.mkdirSync(cmakeDir, {recursive: true}); fs.mkdirSync(project, {recursive: true});
  fs.mkdirSync(path.dirname(targetClang), {recursive: true}); fs.mkdirSync(path.dirname(nativeClang), {recursive: true});
  fs.mkdirSync(path.dirname(header), {recursive: true});
  fs.writeFileSync(targetClang, "target clang fixture\n"); fs.writeFileSync(nativeClang, "native clang fixture\n");
  fs.writeFileSync(header, "OpenCL header fixture\n"); fs.writeFileSync(path.join(path.dirname(header), "opencl-c-base.h"), "OpenCL base fixture\n");
  const originalRule = ["function(generate_pch version)",
    "  add_custom_command(OUTPUT " + cmakeVar("CMAKE_CURRENT_BINARY_DIR") + "/opencl" + cmakeVar("version") + "-c.pch",
    "    COMMAND $<TARGET_FILE:clang> -cc1 -x cl-header -triple amdgcn-amd-amdhsa",
    "    -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL" + cmakeVar("version"),
    "      -emit-pch -o " + cmakeVar("CMAKE_CURRENT_BINARY_DIR") + "/opencl" + cmakeVar("version") + "-c.pch",
    "      < " + cmakeVar("OPENCL_C_H"),
    "    DEPENDS clang " + cmakeVar("OPENCL_C_H") + " " + cmakeVar("clang_resource_headers_gen"),
    "    COMMENT \"Generating opencl" + cmakeVar("version") + "-c.pch\")",
    "  add_custom_target(opencl" + cmakeVar("version") + "-c.pch_target ALL",
    "    DEPENDS " + cmakeVar("CMAKE_CURRENT_BINARY_DIR") + "/opencl" + cmakeVar("version") + "-c.pch)",
    "endfunction()", "generate_pch(1.2)", "generate_pch(2.0)", ""].join("\n");
  fs.writeFileSync(pchRule, originalRule);
  const fullPatch = fs.readFileSync(path.join(repo, "runtime/rocm-comgr-cross-tools.patch"), "utf8"),
    start = fullPatch.indexOf("diff --git a/amd/comgr/cmake/opencl_pch.cmake");
  if (start < 0) {throw new Error("Missing native PCH patch section");}
  fs.writeFileSync(patchFile, fullPatch.slice(start));
  const checked = spawnSync("/usr/bin/git", ["apply", "--check", patchFile], {cwd: source, encoding: "utf8"});
  assert.equal(checked.status, 0, (checked.stdout || "") + (checked.stderr || ""));
  const applied = spawnSync("/usr/bin/git", ["apply", patchFile], {cwd: source, encoding: "utf8"});
  assert.equal(applied.status, 0, (applied.stdout || "") + (applied.stderr || ""));
  fs.writeFileSync(path.join(project, "CMakeLists.txt"), ["cmake_minimum_required(VERSION 3.20)",
    "project(comgr_pch_patch_fixture LANGUAGES NONE)", "message(STATUS \"PCH_FIXTURE_CROSSCOMPILING=" + cmakeVar("CMAKE_CROSSCOMPILING") + "\")",
    "add_executable(clang IMPORTED GLOBAL)", "set_property(TARGET clang PROPERTY IMPORTED_LOCATION \"" + cmakeVar("PCH_TARGET_CLANG") + "\")",
    "set(OPENCL_C_H \"" + cmakeVar("COMGR_OPENCL_C_HEADER") + "\")", "include(\"" + cmakeVar("PCH_RULE") + "\")", ""].join("\n"));
  return {root, project, targetClang, nativeClang, resourceDir, header, pchRule};
}

/** @param {ReturnType<typeof pchPatchFixture>} fixture @param {string} build @param {boolean} cross
 * @param {string} nativeCompiler @param {string} resourceDir @param {string} header */
function configurePchFixture(fixture, build, cross, nativeCompiler, resourceDir, header) {
  const args = ["-S", fixture.project, "-B", build, "-G", "Ninja", "-DPCH_TARGET_CLANG=" + fixture.targetClang,
    "-DPCH_RULE=" + fixture.pchRule, "-DCOMGR_OPENCL_C_HEADER=" + header,
    "-DCOMGR_NATIVE_CLANG_EXECUTABLE=" + nativeCompiler, "-DCOMGR_NATIVE_CLANG_RESOURCE_DIR=" + resourceDir];
  if (cross) {args.push("-DCMAKE_SYSTEM_NAME=Windows");}
  return spawnSync("cmake", args, {encoding: "utf8"});
}

test.describe("Original Windows COMGR static SDK producer", {skip: process.platform !== "linux" && "Linux Windows cross-build plan"}, () => {
  test("uses the official Windows LLVM admission verifier", () => assert.equal(typeof verifyWindowsLlvmAdmission, "function"));

  test("plans directly from the thin Windows LLVM descriptor without controller-only fields", t => {
    const f = fixture(t), thin = {version: 1,
        tools: {clangCl: f.descriptor.tools.clangCl, lldLink: f.descriptor.tools.lldLink,
          llvmNm: f.descriptor.tools.llvmNm, clangResourceInclude: f.descriptor.tools.clangResourceInclude},
        windowsSdk: {overlay: f.descriptor.windowsSdk.overlay, sourceRoot: f.descriptor.windowsSdk.sourceRoot,
          includeRoots: f.descriptor.windowsSdk.includeRoots, libraryRoots: f.descriptor.windowsSdk.libraryRoots,
          msvcLibraries: f.descriptor.windowsSdk.msvcLibraries, ucrtLibraries: f.descriptor.windowsSdk.ucrtLibraries,
          importLibraries: f.descriptor.windowsSdk.importLibraries}},
      thinPlan = windowsComgrSdkPlan({...f.input, descriptor: thin}), fullPlan = windowsComgrSdkPlan(f.input);
    assert.deepEqual(Object.keys(thin).sort(), ["tools", "version", "windowsSdk"]);
    assert.deepEqual(thinPlan.configure, fullPlan.configure);
    assert.deepEqual(thinPlan.definitions, fullPlan.definitions);
    assert.equal(thinPlan.config.tools.lldLink, thin.tools.lldLink);
    assert.equal(thinPlan.config.windowsSdk.overlay, thin.windowsSdk.overlay);
    assert.throws(() => windowsComgrSdkPlan({...f.input, descriptor: {...thin, tools: {...thin.tools, lldLink: undefined}}}),
      /tool lldLink/);
    assert.throws(() => windowsComgrSdkPlan({...f.input, descriptor: {...thin, windowsSdk: {...thin.windowsSdk, libraryRoots: []}}}),
      /Windows SDK library roots/);
  });

  test("resolves LLVM packages through the admitted cache when its receipt has a fresh output root", t => {
    const f = fixture(t), targetLlvmLayout = {source: path.join(f.root, "original-cache", "source"),
        build: path.join(f.root, "original-cache", "build")}, plan = windowsComgrSdkPlan(f.input, targetLlvmLayout);
    assert.equal(plan.targetLlvm, f.input.targetLlvm, "the fresh admitted output remains the identity root");
    assert.equal(plan.targetLlvmSource, targetLlvmLayout.source);
    assert.equal(plan.targetLlvmBuild, targetLlvmLayout.build);
    assert.equal(plan.definitions.LLVM_DIR, path.join(targetLlvmLayout.build, "lib/cmake/llvm"));
    assert.equal(plan.definitions.Clang_DIR, path.join(targetLlvmLayout.build, "lib/cmake/clang"));
    assert.equal(plan.definitions.LLD_DIR, path.join(targetLlvmLayout.build, "lib/cmake/lld"));
    assert.ok(plan.configure.includes("-DLLVM_DIR=" + path.join(targetLlvmLayout.build, "lib/cmake/llvm")));
    assert.equal(windowsComgrSdkPlan(f.input).targetLlvmBuild, path.join(f.input.targetLlvm, "build"),
      "ordinary non-refresh admissions keep their existing layout");
  });

  test("pins Git itself and disables replacement refs for pinned source selection", t => {
    const f = fixture(t), plan = windowsComgrSdkPlan(f.input), source = path.join(f.root, "replace-ref-repository"),
      contents = path.join(source, "amd/comgr/source.txt"), marker = path.join(f.root, "shadow-git-ran"), replacementArchive = path.join(f.root, "replacement.tar");
    assert.equal(plan.git, "/usr/bin/git");
    assert.equal(plan.archiveCommand[0], plan.git);
    assert.ok(plan.applyCommands.every(command => command[0] === plan.git));
    fs.mkdirSync(path.dirname(contents), {recursive: true});
    const setupEnv = /** @type {NodeJS.ProcessEnv} */ ({...process.env, GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null"});
    /** @param {string[]} args */
    function setupGit(args) {
      const result = spawnSync(plan.git, args, {encoding: "utf8", env: setupEnv});
      assert.equal(result.status, 0, (result.stdout || "") + (result.stderr || ""));
      return (result.stdout || "").trim();
    }
    setupGit(["init", "--quiet", source]);
    setupGit(["-C", source, "config", "user.name", "fixture"]);
    setupGit(["-C", source, "config", "user.email", "fixture@example.invalid"]);
    fs.writeFileSync(contents, "original source\n");
    setupGit(["-C", source, "add", "amd/comgr/source.txt"]);
    setupGit(["-C", source, "commit", "--quiet", "-m", "original"]);
    const original = setupGit(["-C", source, "rev-parse", "HEAD"]);
    fs.writeFileSync(contents, "replacement source\n");
    setupGit(["-C", source, "commit", "--quiet", "-am", "replacement"]);
    const replacement = setupGit(["-C", source, "rev-parse", "HEAD"]);
    setupGit(["-C", source, "replace", original, replacement]);

    fs.mkdirSync(plan.nativeBin, {recursive: true});
    const shadowGit = path.join(plan.nativeBin, "git");
    fs.writeFileSync(shadowGit, "#!/bin/sh\nprintf used > " + JSON.stringify(marker) + "\nexit 97\n");
    fs.chmodSync(shadowGit, 0o755);
    const env = cleanEnvironment({}, plan, plan.nativeBin);
    assert.equal(env["GIT_NO_REPLACE_OBJECTS"], "1");
    assert.equal((env["PATH"] || "").split(path.delimiter)[0], plan.nativeBin);
    assert.equal(spawnSync("git", ["--version"], {encoding: "utf8", env}).status, 97, "fixture shadow must be first in PATH");
    assert.equal(fs.existsSync(marker), true, "the PATH probe must execute the fixture shadow");
    fs.unlinkSync(marker);

    const archiveArgs = [...plan.archiveCommand.slice(1)], checkout = archiveArgs.indexOf("-C");
    archiveArgs[checkout + 1] = source;
    archiveArgs[archiveArgs.length - 1] = original + ":amd/comgr";
    archiveArgs[archiveArgs.indexOf("--output=" + plan.archive)] = "--output=" + replacementArchive;
    const unprotectedEnv = {...setupEnv};
    delete unprotectedEnv["GIT_NO_REPLACE_OBJECTS"];
    const unprotected = spawnSync(plan.git, archiveArgs, {cwd: repo, env: unprotectedEnv, encoding: "utf8"});
    assert.equal(unprotected.status, 0, (unprotected.stdout || "") + (unprotected.stderr || ""));
    const changed = spawnSync("/usr/bin/tar", ["-xOf", replacementArchive, "amd/comgr/source.txt"], {encoding: "utf8"});
    assert.equal(changed.status, 0);
    assert.equal(changed.stdout, "replacement source\n", "fixture replacement ref must alter the unprotected archive");

    const protectedArchive = path.join(f.root, "pinned.tar");
    archiveArgs[archiveArgs.indexOf("--output=" + replacementArchive)] = "--output=" + protectedArchive;
    const protectedResult = spawnSync(plan.git, archiveArgs, {cwd: repo, env, encoding: "utf8"});
    assert.equal(protectedResult.status, 0, (protectedResult.stdout || "") + (protectedResult.stderr || ""));
    const originalContents = spawnSync("/usr/bin/tar", ["-xOf", protectedArchive, "amd/comgr/source.txt"], {encoding: "utf8"});
    assert.equal(originalContents.status, 0);
    assert.equal(originalContents.stdout, "original source\n");
    assert.equal(fs.existsSync(marker), false, "absolute Git calls must not execute a PATH shadow");
  });

  test("does not fingerprint the unrelated ACPP SDK producer", () => {
    assert.ok(!recipeFiles.includes("scripts/build-native-windows-acpp-sdk.js"));
  });

  test("plans the original static target with independent native and Windows Intel compilers", t => {
    const f = fixture(t), plan = windowsComgrSdkPlan(f.input);
    assert.equal(plan.revision, revision);
    assert.deepEqual(plan.patches, ["runtime/rocm-comgr-llvm21.patch", "runtime/rocm-comgr-resources.patch", "runtime/rocm-comgr-cross-tools.patch"]);
    assert.equal(plan.targetTriple, "x86_64-pc-windows-msvc");
    assert.equal(plan.compilerTools.CMAKE_C_COMPILER, f.descriptor.tools.clangCl);
    assert.equal(plan.compilerTools.CMAKE_CXX_COMPILER, f.descriptor.tools.clangCl);
    assert.equal(plan.hostConfigure.includes("-DCMAKE_C_COMPILER=" + f.input.hostCompiler), true);
    assert.equal(plan.hostHelper, path.join(plan.hostBuild, "bc2h"));
    assert.equal(plan.definitions.COMGR_BC2H_EXECUTABLE, plan.hostHelper);
    assert.equal(plan.definitions.COMGR_BUILD_SHARED_LIBS, "OFF");
    assert.equal(plan.definitions.COMGR_EMBEDDED_RESOURCES, "ON");
    assert.equal(plan.definitions.LLVM_DIR, path.join(f.input.targetLlvm, "build/lib/cmake/llvm"));
    assert.equal(plan.definitions.Clang_DIR, path.join(f.input.targetLlvm, "build/lib/cmake/clang"));
    assert.equal(plan.definitions.LLD_DIR, path.join(f.input.targetLlvm, "build/lib/cmake/lld"));
    assert.equal(plan.definitions.AMDDeviceLibs_DIR, path.join(f.input.deviceInstall, "lib/cmake/AMDDeviceLibs"));
    assert.equal(plan.native.clang, path.join(f.input.nativeLlvm, "bin/clang"));
    assert.deepEqual(plan.nativeRuntimeRoots, [path.join(f.input.nativeLlvm, "lib")]);
    assert.ok(!plan.runtimeLibraryRoots.includes("/opt/ubuntu24-libs"));
    assert.equal(plan.nativeClangResourceDir, path.join(f.input.nativeLlvm, "lib/clang/21"));
    assert.equal(plan.openclHeader, path.join(plan.nativeClangResourceDir, "include/opencl-c.h"));
    assert.equal(plan.openclBaseHeader, path.join(plan.nativeClangResourceDir, "include/opencl-c-base.h"));
    assert.equal(plan.definitions.COMGR_NATIVE_CLANG_EXECUTABLE, plan.native.clang);
    assert.equal(plan.definitions.COMGR_NATIVE_CLANG_RESOURCE_DIR, plan.nativeClangResourceDir);
    assert.equal(plan.definitions.COMGR_OPENCL_C_HEADER, plan.openclHeader);
    assert.equal(plan.definitions.CMAKE_TRY_COMPILE_CONFIGURATION, "Release");
    assert.match(plan.ruleOverrideText, /set\(CMAKE_C_FLAGS_RELEASE_INIT \[==\[\/O2 \/DNDEBUG\]==\]\)/);
    assert.match(plan.ruleOverrideText, /set\(CMAKE_CXX_FLAGS_RELEASE_INIT \[==\[\/O2 \/DNDEBUG\]==\]\)/);
    assert.notEqual(plan.openclHeader, path.join(f.descriptor.tools.clangResourceInclude, "opencl-c.h"));
    assert.match(plan.toolchain, /CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY/);
    assert.match(plan.toolchain, /CMAKE_MSVC_RUNTIME_LIBRARY/);
    assert.match(plan.toolchain, /CMAKE_INCLUDE_SYSTEM_FLAG_C .*external:I/);
    assert.match(plan.toolchain, /CMAKE_INCLUDE_SYSTEM_FLAG_CXX .*external:I/);
    assert.match(plan.toolchain, /-nostdinc/);
    assert.match(plan.toolchain, /"\/MT"/);
    assert.match(plan.toolchain, /"\/EHsc"/);
    assert.match(plan.toolchain, /"\/GR"/);
    assert.match(plan.hostProjectText, /cmake\/bc2h\.cmake/);
    assert.doesNotMatch(plan.toolchain, /CROSSCOMPILING_EMULATOR/);
    assert.deepEqual(hostSources, ["src/comgr-cache.cpp", "src/comgr-cache-command.cpp", "src/comgr-cache-bundler-command.cpp",
      "src/comgr-clang-command.cpp", "src/comgr-compiler.cpp", "src/comgr.cpp", "src/comgr-device-libs.cpp",
      "src/comgr-diagnostic-handler.cpp", "src/comgr-disassembly.cpp", "src/comgr-elfdump.cpp", "src/comgr-env.cpp",
      "src/comgr-metadata.cpp", "src/comgr-objdump.cpp", "src/comgr-signal.cpp", "src/comgr-spirv-command.cpp",
      "src/comgr-symbol.cpp", "src/comgr-symbolizer.cpp", "src/time-stat/time-stat.cpp"]);
  });

  test("retains actual pinned native runtime libraries and rejects escaping shared libraries", t => {
    const f = fixture(t), plan = windowsComgrSdkPlan(f.input), env = cleanEnvironment({LD_LIBRARY_PATH: "/attacker/loader"}, plan, plan.nativeBin);
    assert.equal(env["LD_LIBRARY_PATH"], plan.runtimeLibraryRoots.join(path.delimiter));
    assert.ok(!env["LD_LIBRARY_PATH"].includes("/opt/ubuntu24-libs"));
    assert.ok(!env["LD_LIBRARY_PATH"].includes("/attacker/loader"));

    const runtime = path.join(f.root, "runtime"), inside = path.join(runtime, "libLLVM.so.21.1"), outside = path.join(f.root, "outside.so");
    fs.mkdirSync(runtime); fs.writeFileSync(inside, "pinned runtime"); fs.writeFileSync(outside, "foreign runtime");
    fs.symlinkSync("missing.so", path.join(runtime, "liboptional-alias.so"));
    assert.deepEqual(nativeRuntimeInputs({...plan, nativeRuntimeRoots: [runtime]}), [inside]);
    const loop = path.join(runtime, "libloop.so"); fs.symlinkSync("libloop.so", loop);
    assert.throws(() => nativeRuntimeInputs({...plan, nativeRuntimeRoots: [runtime]}),
      error => error instanceof Error && /** @type {NodeJS.ErrnoException} */ (error).code === "ELOOP");
    fs.unlinkSync(loop);
    fs.symlinkSync(outside, path.join(runtime, "libescape.so"));
    assert.throws(() => nativeRuntimeInputs({...plan, nativeRuntimeRoots: [runtime]}), /escapes its selected root/);
    const empty = path.join(f.root, "empty-runtime"); fs.mkdirSync(empty);
    fs.symlinkSync("missing.so", path.join(empty, "liboptional-alias.so"));
    assert.throws(() => nativeRuntimeInputs({...plan, nativeRuntimeRoots: [empty]}), /no available shared libraries/);

    const emptySdk = path.join(f.root, "empty-sdk-lib"); fs.mkdirSync(emptySdk);
    assert.deepEqual(nativeRuntimeInputs({...plan, nativeRuntimeRoots: [emptySdk]}), [],
      "a canonical SDK lib directory may have no private DSOs when tools use the host OS runtime");
    const sdkAlias = path.join(f.root, "empty-sdk-alias"); fs.symlinkSync(emptySdk, sdkAlias);
    assert.throws(() => nativeRuntimeInputs({...plan, nativeRuntimeRoots: [sdkAlias]}), /canonical directory/);
    assert.throws(() => nativeRuntimeInputs({...plan, nativeRuntimeRoots: []}), /runtime roots are empty/);
  });

  test("binds unsupported llvm-lib mode only to byte-identical version-checked llvm-ar", t => {
    const plan = windowsComgrSdkPlan(fixture(t).input), root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-comgr-librarian-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    assert.equal(plan.nativeQueries["llvm-lib"], undefined);
    assert.deepEqual(plan.nativeQueries["llvm-ar"], [plan.native["llvm-ar"], ["--version"]]);
    const llvmLib = path.join(root, "llvm-lib"), llvmAr = path.join(root, "llvm-ar"), version = "Ubuntu LLVM version 21.1.8\n  Optimized build.\n";
    fs.writeFileSync(llvmLib, "pinned llvm-ar bytes"); fs.writeFileSync(llvmAr, "pinned llvm-ar bytes");
    assert.deepEqual(verifyNativeLibrarianIdentity(llvmLib, llvmAr, version), {version, sha256: fileHash(llvmAr)});
    fs.writeFileSync(llvmLib, "different archive tool bytes");
    assert.throws(() => verifyNativeLibrarianIdentity(llvmLib, llvmAr, version), /byte-identical/);
    assert.throws(() => verifyNativeLibrarianIdentity(llvmLib, llvmAr, "LLVM version 20.1.0"), /version-checked LLVM 21\.1\.8/);
  });

  test("the cross-only bc2h patch keeps native generation intact and requires an explicit host tool", () => {
    const patch = fs.readFileSync(path.join(repo, "runtime/rocm-comgr-cross-tools.patch"), "utf8");
    assert.match(patch, /^@@ -25,7 \+25,17 @@$/m);
    assert.match(patch, /if\(CMAKE_CROSSCOMPILING\)/);
    assert.match(patch, /NOT IS_ABSOLUTE "\$\{COMGR_BC2H_EXECUTABLE\}"/);
    assert.match(patch, /add_executable\(bc2h IMPORTED GLOBAL\)/);
    assert.match(patch, /add_executable\(bc2h \$\{CMAKE_CURRENT_BINARY_DIR\}\/bc2h\.c\)/);
    assert.match(patch, /COMGR_NATIVE_CLANG_RESOURCE_DIR}\/include\/opencl-c-base\.h/);
    assert.match(patch, /set\(clang_pch_compiler "\$\{COMGR_NATIVE_CLANG_EXECUTABLE\}"\)/);
    assert.match(patch, /set\(clang_pch_compiler "\$<TARGET_FILE:clang>"\)/);
    assert.match(patch, /-cc1 \$\{clang_pch_resource_dir\} -x cl-header -triple amdgcn-amd-amdhsa/);
    assert.doesNotMatch(patch, /CROSSCOMPILING_EMULATOR|fake|emulat/i);
  });

  test("routes cross PCH generation through native LLVM 21 and leaves native clang selection unchanged", t => {
    const cmake = spawnSync("cmake", ["--version"], {encoding: "utf8"}), ninja = spawnSync("ninja", ["--version"], {encoding: "utf8"});
    if (cmake.error || cmake.status !== 0 || ninja.error || ninja.status !== 0) {return t.skip("CMake and Ninja are required for PCH branch checks");}
    const f = pchPatchFixture(t), native = configurePchFixture(f, path.join(f.root, "native-build"), false, f.nativeClang, f.resourceDir, f.header);
    assert.equal(native.status, 0, (native.stdout || "") + (native.stderr || ""));
    assert.match(native.stdout, /PCH_FIXTURE_CROSSCOMPILING=FALSE/);
    const nativeNinja = fs.readFileSync(path.join(f.root, "native-build/build.ninja"), "utf8");
    assert.ok(nativeNinja.includes(f.targetClang.replaceAll(" ", "$ ")));
    assert.doesNotMatch(nativeNinja, /-resource-dir/);
    assert.ok(!nativeNinja.includes(f.nativeClang.replaceAll(" ", "$ ")));

    const cross = configurePchFixture(f, path.join(f.root, "cross-build"), true, f.nativeClang, f.resourceDir, f.header);
    assert.equal(cross.status, 0, (cross.stdout || "") + (cross.stderr || ""));
    assert.match(cross.stdout, /PCH_FIXTURE_CROSSCOMPILING=TRUE/);
    const crossNinja = fs.readFileSync(path.join(f.root, "cross-build/build.ninja"), "utf8");
    assert.ok(crossNinja.includes(f.nativeClang.replaceAll(" ", "$ ")));
    assert.match(crossNinja, /-cc1 -resource-dir .*lib\/clang\/21 -x cl-header -triple amdgcn-amd-amdhsa/);
    assert.match(crossNinja, /-Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL1\.2/);
    assert.ok(!crossNinja.includes(f.targetClang.replaceAll(" ", "$ ")));

    const relative = configurePchFixture(f, path.join(f.root, "relative-build"), true, "relative/clang", f.resourceDir, f.header);
    assert.notEqual(relative.status, 0);
    assert.match((relative.stdout || "") + (relative.stderr || ""), /explicit native LLVM 21 Clang resources/);
    const missing = configurePchFixture(f, path.join(f.root, "missing-build"), true, path.join(f.root, "absent-clang"), f.resourceDir, f.header);
    assert.notEqual(missing.status, 0);
    assert.match((missing.stdout || "") + (missing.stderr || ""), /explicit native LLVM 21 Clang resources/);
    const relativeResource = configurePchFixture(f, path.join(f.root, "relative-resource-build"), true, f.nativeClang, "relative/resource", f.header);
    assert.notEqual(relativeResource.status, 0);
    assert.match((relativeResource.stdout || "") + (relativeResource.stderr || ""), /explicit native LLVM 21 Clang resources/);
    const foreignHeader = path.join(f.root, "Intel 2026.2/include/opencl-c.h");
    fs.mkdirSync(path.dirname(foreignHeader), {recursive: true}); fs.writeFileSync(foreignHeader, "unselected OpenCL header\n");
    const wrongHeader = configurePchFixture(f, path.join(f.root, "wrong-header-build"), true, f.nativeClang, f.resourceDir, foreignHeader);
    assert.notEqual(wrongHeader.status, 0);
    assert.match((wrongHeader.stdout || "") + (wrongHeader.stderr || ""), /must use the native LLVM 21 OpenCL-C header/);
  });

  test("rejects non-LLVM-21 native PCH compilers or a mismatched resource root", t => {
    const plan = windowsComgrSdkPlan(fixture(t).input), valid = "Ubuntu clang version 21.1.8 (test build)";
    assert.deepEqual(verifyNativeClangIdentity(valid, plan.nativeClangResourceDir, plan), {version: "21.1.8", resourceDir: plan.nativeClangResourceDir});
    for (const version of ["Ubuntu clang version 20.1.0", "Intel(R) clang version 23.0.1", "clang version 21.2.0"]) {
      assert.throws(() => verifyNativeClangIdentity(version, plan.nativeClangResourceDir, plan), /LLVM 21\.1\.8/);
    }
    assert.throws(() => verifyNativeClangIdentity(valid, "./lib/clang/21", plan), /resource directory/);
  });

  test("admits only the canonical native clang executable or its exact clang-21 package alias", t => {
    const root = tempDir(t, "mom-native-clang-alias-");
    /** @param {string} name */
    const nativeRoot = (/** @type {string} */ name) => path.join(root, name),
      prepare = (/** @type {string} */ name, /** @type {string|null} */ aliasTarget,
        /** @type {"regular"|"chain"} */ targetKind = "regular", /** @type {boolean} */ executable = true) => {
        const selectedRoot = nativeRoot(name), bin = path.join(selectedRoot, "bin"); fs.mkdirSync(bin, {recursive: true});
        if (targetKind === "regular") {
          const target = path.join(bin, "clang-21"); fs.writeFileSync(target, "native clang fixture\n"); fs.chmodSync(target, executable ? 0o755 : 0o644);
        } else if (targetKind === "chain") {
          fs.writeFileSync(path.join(bin, "clang-real"), "native clang fixture\n"); fs.chmodSync(path.join(bin, "clang-real"), 0o755);
          fs.symlinkSync("clang-real", path.join(bin, "clang-21"));
        }
        if (aliasTarget === null) {
          fs.writeFileSync(path.join(bin, "clang"), "native clang fixture\n"); fs.chmodSync(path.join(bin, "clang"), executable ? 0o755 : 0o644);
        } else {fs.symlinkSync(aliasTarget, path.join(bin, "clang"));}
        return {root: selectedRoot, clang: path.join(bin, "clang")};
      };
    const regular = prepare("regular", null);
    assert.equal(canonicalNativeClang(regular.clang, regular.root), regular.clang);
    const alias = prepare("alias", "clang-21");
    assert.equal(canonicalNativeClang(alias.clang, alias.root), alias.clang, "preserve the selected clang argv[0]");

    const outside = path.join(root, "outside", "clang-21"); fs.mkdirSync(path.dirname(outside), {recursive: true});
    fs.writeFileSync(outside, "outside compiler\n"); fs.chmodSync(outside, 0o755);
    for (const [name, target] of /** @type {Array<[string,string]>} */ ([
      ["relative-escape", "../../outside/clang-21"], ["absolute-escape", outside], ["wrong-name", "clang-22"],
    ])) {
      const invalid = prepare(name, target);
      assert.throws(() => canonicalNativeClang(invalid.clang, invalid.root), /exact relative clang -> clang-21/);
    }
    const chained = prepare("chained", "clang-21", "chain");
    assert.throws(() => canonicalNativeClang(chained.clang, chained.root), /exact relative clang -> clang-21/);
    const nonExecutable = prepare("non-executable", "clang-21", "regular", false);
    assert.throws(() => canonicalNativeClang(nonExecutable.clang, nonExecutable.root));
    const rootAlias = path.join(root, "root-alias"); fs.symlinkSync(alias.root, rootAlias);
    assert.throws(() => canonicalNativeClang(path.join(rootAlias, "bin/clang"), rootAlias), /root is not canonical/);
  });

  test("CMake parses the generated toolchain literally, including a hostile-looking SDK path", t => {
    const plan = windowsComgrSdkPlan(fixture(t).input), probe = spawnSync("cmake", ["--version"], {encoding: "utf8"});
    if (probe.error || probe.status !== 0) {return t.skip("CMake unavailable for the toolchain syntax check");}
    const file = path.join(path.dirname(plan.output), "toolchain-test.cmake"), rules = path.join(path.dirname(plan.output), "rules-test.cmake");
    fs.writeFileSync(file, plan.toolchain); fs.writeFileSync(rules,
      'set(CMAKE_C_COMPILER_ID "IntelLLVM")\nset(CMAKE_CXX_COMPILER_ID "IntelLLVM")\n' + plan.ruleOverrideText);
    const result = spawnSync("cmake", ["-P", file], {encoding: "utf8"});
    assert.equal(result.status, 0, (result.stdout || "") + (result.stderr || ""));
    const ruleResult = spawnSync("cmake", ["-P", rules], {encoding: "utf8"});
    assert.equal(ruleResult.status, 0, (ruleResult.stdout || "") + (ruleResult.stderr || ""));
  });

  test("binds the static Intel 2026.2 CMake compiler identities", t => {
    const root = tempDir(t, "mom-windows-comgr-compiler-");
    const build = path.join(root, "build"), files = path.join(build, "CMakeFiles", "3.31.8"); fs.mkdirSync(files, {recursive: true});
    for (const language of ["C", "CXX"]) {
      fs.writeFileSync(path.join(files, "CMake" + language + "Compiler.cmake"),
        `set(CMAKE_${language}_COMPILER_ID "IntelLLVM")\nset(CMAKE_${language}_COMPILER_VERSION "2026.2.0")\n`);
    }
    assert.deepEqual(Object.keys(verifyIntelCompilerIdentity(build)), ["c", "cxx"]);
    fs.writeFileSync(path.join(files, "CMakeCXXCompiler.cmake"), 'set(CMAKE_CXX_COMPILER_ID "Clang")\nset(CMAKE_CXX_COMPILER_VERSION "21.1.8")\n');
    assert.throws(() => verifyIntelCompilerIdentity(build), /IntelLLVM 2026\.2/);
  });

  test("requires the native bc2h C TU and object to be Intel Linux outputs", t => {
    const root = tempDir(t, "mom-native-bc2h-");
    const build = path.join(root, "build"), files = path.join(build, "CMakeFiles", "3.31.8"), compiler = path.join(root, "intel", "bin", "icx");
    fs.mkdirSync(files, {recursive: true});
    fs.writeFileSync(path.join(files, "CMakeCCompiler.cmake"), `set(CMAKE_C_COMPILER_ID "IntelLLVM")\nset(CMAKE_C_COMPILER_VERSION "2026.2.0")\nset(CMAKE_C_COMPILER "${compiler}")\n`);
    assert.equal(verifyNativeIntelCompilerIdentity(build, compiler).version, "2026.2.0");
    const generated = path.join(build, "bc2h.c"), object = path.join(build, "CMakeFiles/bc2h.dir/bc2h.c.o"), commands = path.join(build, "compile_commands.json");
    fs.mkdirSync(path.dirname(object), {recursive: true}); fs.writeFileSync(generated, "int main(void) { return 0; }\n");
    fs.writeFileSync(object, Buffer.from([0x7f, 0x45, 0x4c, 0x46, ...Array(28).fill(0)]));
    writeJson(commands, [{file: generated, directory: build, output: object, command: `"${compiler}" -O2 -c "${generated}" -o "${object}"`}]);
    assert.equal(auditBc2hCompileCommands(commands, compiler, build, generated, object).sha256Object, fileHash(object));
    fs.writeFileSync(commands, JSON.stringify([{file: generated, directory: build, output: object,
      command: `"${compiler}" --target=x86_64-pc-windows-msvc -c "${generated}" -o "${object}"`}]));
    assert.throws(() => auditBc2hCompileCommands(commands, compiler, build, generated, object), /Intel Linux/);
  });

  test("verifies CMake cache ABI, explicit SDKs and native tablegen roots", t => {
    const plan = windowsComgrSdkPlan(fixture(t).input), cache = path.join(path.dirname(plan.output), "CMakeCache.txt");
    const expected = {CMAKE_SYSTEM_NAME: "Windows", CMAKE_C_COMPILER: plan.compilerTools.CMAKE_C_COMPILER,
      CMAKE_CXX_COMPILER: plan.compilerTools.CMAKE_CXX_COMPILER, CMAKE_C_COMPILER_TARGET: plan.targetTriple,
      CMAKE_CXX_COMPILER_TARGET: plan.targetTriple, CMAKE_LINKER: plan.compilerTools.CMAKE_LINKER,
      CMAKE_AR: plan.compilerTools.CMAKE_AR, CMAKE_RANLIB: plan.compilerTools.CMAKE_RANLIB,
      CMAKE_INCLUDE_SYSTEM_FLAG_C: "/external:I", CMAKE_INCLUDE_SYSTEM_FLAG_CXX: "/external:I",
      CMAKE_MAKE_PROGRAM: plan.tools.ninja, CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded", CMAKE_BUILD_TYPE: "Release",
      CMAKE_C_FLAGS_RELEASE: "/O2 /DNDEBUG", CMAKE_CXX_FLAGS_RELEASE: "/O2 /DNDEBUG", CMAKE_TRY_COMPILE_CONFIGURATION: "Release",
      COMGR_BUILD_SHARED_LIBS: "OFF", COMGR_EMBEDDED_RESOURCES: "ON", COMGR_DISABLE_SPIRV: "ON",
      COMGR_BC2H_EXECUTABLE: plan.hostHelper, LLVM_LINK_LLVM_DYLIB: "OFF", LLVM_DIR: plan.llvmDir, Clang_DIR: plan.clangDir,
      LLD_DIR: plan.lldDir, AMDDeviceLibs_DIR: plan.deviceConfig, COMGR_OPENCL_C_HEADER: plan.openclHeader,
      COMGR_NATIVE_CLANG_EXECUTABLE: plan.native.clang, COMGR_NATIVE_CLANG_RESOURCE_DIR: plan.nativeClangResourceDir,
      CMAKE_EXPORT_COMPILE_COMMANDS: "ON", LLVM_NATIVE_TOOL_DIR: plan.nativeBin, LLVM_TABLEGEN: plan.native["llvm-tblgen"],
      CLANG_TABLEGEN: plan.native["clang-tblgen"], CMAKE_TRY_COMPILE_TARGET_TYPE: "STATIC_LIBRARY",
      CMAKE_DISABLE_FIND_PACKAGE_ROCM: "ON", CMAKE_DISABLE_FIND_PACKAGE_HIP: "ON", CMAKE_DISABLE_FIND_PACKAGE_OpenCL: "ON"};
    fs.writeFileSync(cache, Object.entries(expected).map(([name, value]) => name + ":STRING=" + value).join("\n") + "\n");
    assert.equal(verifyComgrCache(cache, plan), cache);
    fs.writeFileSync(cache, fs.readFileSync(cache, "utf8").replace("COMGR_EMBEDDED_RESOURCES:STRING=ON", "COMGR_EMBEDDED_RESOURCES:STRING=OFF"));
    assert.throws(() => verifyComgrCache(cache, plan), /COMGR_EMBEDDED_RESOURCES/);
  });

  test("audits exactly 18 original /MT /EHsc /GR Intel COFF host TUs", t => {
    const root = tempDir(t, "mom-comgr-compile-audit-");
    const source = path.join(root, "source"), build = path.join(root, "build"), objectRoot = path.join(build, "CMakeFiles/amd_comgr.dir"),
      compiler = path.join(root, "Intel 2026.2", "bin", "icx-cl"), commands = path.join(build, "compile_commands.json");
    fs.mkdirSync(path.join(source, "src/time-stat"), {recursive: true}); fs.mkdirSync(objectRoot, {recursive: true});
    const entries = hostSources.map(name => {
      const file = path.join(source, name), output = path.join(objectRoot, name + ".obj");
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.mkdirSync(path.dirname(output), {recursive: true}); fs.writeFileSync(file, "void fixture() {}\n");
      return {file, directory: build, output,
        command: `"${compiler}" --target=x86_64-pc-windows-msvc /MT /EHsc /GR /c "${file}" /Fo"${output}"`};
    });
    const first = entries[0];
    assert.ok(first);
    writeJson(commands, entries);
    assert.equal(auditComgrCompileCommands(commands, compiler, source, build, false).count, 18);
    for (const entry of entries) {fs.writeFileSync(entry.output, Buffer.from([0x64, 0x86, ...Array(30).fill(0)]));}
    const audited = auditComgrCompileCommands(commands, compiler, source, build, true);
    assert.equal(audited.objects.length, 18);
    const archiveMembers = audited.objects.map(row => path.relative(build, row.output).split(path.sep).join("/"));
    assert.deepEqual(auditComgrArchiveMembers(archiveMembers.join("\n"), audited.objects, build).sort(), archiveMembers.sort());
    assert.throws(() => auditComgrArchiveMembers("../amd_comgr.obj", audited.objects, build), /exactly its 18 original COFF/);
    first.command += " /MT"; writeJson(commands, entries);
    assert.equal(auditComgrCompileCommands(commands, compiler, source, build, true).count, 18);
    first.command = first.command.replace(/ \/MT$/, ""); writeJson(commands, entries);
    first.command = first.command.replace("/MT", "/MD"); writeJson(commands, entries);
    assert.throws(() => auditComgrCompileCommands(commands, compiler, source, build, true), /wrong compiler, ABI/);
    first.command = first.command.replace("/MD", "/MT"); writeJson(commands, entries);
    first.command = first.command.replace("/MT", "-MD"); writeJson(commands, entries);
    assert.throws(() => auditComgrCompileCommands(commands, compiler, source, build, true), /wrong compiler, ABI/);
    first.command = first.command.replace("-MD", "/MT"); writeJson(commands, entries);
    first.command += " -isystem /usr/include"; writeJson(commands, entries);
    assert.throws(() => auditComgrCompileCommands(commands, compiler, source, build, true), /wrong compiler, ABI/);
    first.command = first.command.replace(" -isystem /usr/include", ""); writeJson(commands, entries);
    fs.writeFileSync(first.output, Buffer.from([0x4c, 0x01, ...Array(30).fill(0)]));
    assert.throws(() => auditComgrCompileCommands(commands, compiler, source, build, true), /x64 COFF/);
  });

  test("validates the preserved device receipt, 71 bitcode files and generated registry", t => {
    const f = deviceReceiptFixture(t), proof = verifyWindowsDeviceResources(f.receiptFile, f.install);
    assert.equal(proof.files.length, 72);
    assert.equal(proof.device[f.configFile], fileHash(f.configFile));
    fs.writeFileSync(path.join(f.bitcode, "untracked.bc"), "extra\n");
    assert.throws(() => verifyWindowsDeviceResources(f.receiptFile, f.install), /71-file set/);
    fs.unlinkSync(path.join(f.bitcode, "untracked.bc"));
    fs.writeFileSync(path.join(f.bitcode, "ocml.bc"), "changed\n");
    assert.throws(() => verifyWindowsDeviceResources(f.receiptFile, f.install), /bytes differ/);
  });

  test("discovers the original amd_comgr CMake target without copying it to a prefix", t => {
    const root = tempDir(t, "mom-comgr-file-api-");
    const source = path.join(root, "source"), build = path.join(root, "build"), reply = path.join(build, ".cmake/api/v1/reply");
    fs.mkdirSync(source); fs.mkdirSync(reply, {recursive: true});
    writeJson(path.join(reply, "index-test.json"), {reply: {"codemodel-v2": {jsonFile: "codemodel.json"}}});
    writeJson(path.join(reply, "codemodel.json"), {paths: {source, build}, configurations: [{targets: [{id: "unexpected-id", jsonFile: "target.json"}]}]});
    writeJson(path.join(reply, "target.json"), {name: "amd_comgr", type: "STATIC_LIBRARY", artifacts: [{path: "lib/amd_comgr_7.lib"}]});
    assert.equal(comgrArchive(build, source).archive, path.join(build, "lib/amd_comgr_7.lib"));
    fs.writeFileSync(path.join(reply, "target.json"), JSON.stringify({name: "amd_comgr", type: "EXECUTABLE", artifacts: [{path: "bin/amd_comgr.exe"}]}));
    assert.throws(() => comgrArchive(build, source), /original static COFF/);
  });
});
