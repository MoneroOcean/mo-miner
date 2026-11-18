"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {compilerArchives} = require("../scripts/native-compiler-isolation");
const {jitSdkPlan} = require("../scripts/native-jit-sdk");
const {sdkComponents} = require("../scripts/prepare-native-sdk-sources");

const root = path.join(__dirname, "..");
const cmake = fs.readFileSync(path.join(root, "tests/native/dpcpp-jit/CMakeLists.txt"), "utf8");
const nativeJitSource = fs.readFileSync(path.join(root, "tests/reference/native-dpcpp-jit.cpp"), "utf8");
const nativeSourceEmitter = fs.readFileSync(path.join(root, "tests/reference/native-kawpow-source.cpp"), "utf8");
const staticJitPatch = fs.readFileSync(path.join(root, "runtime/dpcpp-static-jit.patch"), "utf8").replace(/\r\n/g, "\n");
const sourceManifest = [
  "tests/native/dpcpp-jit/CMakeLists.txt",
  "tests/reference/native-dpcpp-jit.cpp",
  "tests/reference/native-kawpow-source.cpp",
  "runtime/driver-library.cpp",
  "scripts/native-resources.js",
  "sycl/kawpow/device.inc",
  "sycl/kawpow/keccak.inc",
];
const windowsSystemLibraries = ["psapi", "shell32", "ole32", "uuid", "advapi32", "ws2_32", "ntdll", "shlwapi"];
const linuxLinkFlags = ["-Wl,--gc-sections", "-Wl,--wrap=dlopen", "-Wl,--wrap=dlerror", "-Wl,--wrap=dlsym,--wrap=dlvsym", "-static-libstdc++", "-static-libgcc"];

test.describe("Original static compiler SDK planning", {skip: process.platform !== "linux"}, () => {
  const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/inputs/" + name]));
  const tools = {cmake: "/tools/cmake", ninja: "/tools/ninja"};
  const plan = () => jitSdkPlan("linux", roots, "/compiler", "/resources", "/output", tools, 2);
  test("uses the original source graph and all maintained hardware compiler targets", () => {
    const p = plan(); assert.equal(p.definitions["LLVM_TARGETS_TO_BUILD"], "X86;NVPTX;SPIRV");
    assert.deepEqual(p.configure.slice(0, 5), [tools.cmake, "-S", "/inputs/dpcpp/llvm", "-B", "/output/build"]);
  });
  test("registers the translator before libdevice and attaches the real closure proof last", () => {
    const d = plan().definitions;
    assert.equal(d["LLVM_EXTERNAL_PROJECTS"], "sycl;llvm-spirv;libdevice;sycl-jit;mom-jit-proof");
    assert.equal(d["LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR"], "/inputs/dpcpp/llvm-spirv");
    assert.equal(d["LLVM_EXTERNAL_MOM_JIT_PROOF_SOURCE_DIR"], path.join(root, "tests/native/dpcpp-jit"));
    assert.ok(!d["LLVM_ENABLE_PROJECTS"]?.includes("mom-jit-proof"), "External proof is not an upstream sibling project");
  });
  test("supplies both original external dependencies and rejects empty disconnected fetches", () => {
    const d = plan().definitions;
    assert.equal(d["FETCHCONTENT_SOURCE_DIR_VC-INTRINSICS"], roots["vcIntrinsics"]);
    assert.equal(d["LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR"], roots["spirvHeaders"]);
    assert.equal(d["FETCHCONTENT_FULLY_DISCONNECTED"], "ON"); assert.equal(d["CMAKE_POLICY_DEFAULT_CMP0170"], "NEW");
  });
  test("keeps static runtime semantics and the in-process Clang frontend", () => {
    const d = plan().definitions;
    for (const name of ["BUILD_SHARED_LIBS", "LLVM_BUILD_LLVM_DYLIB", "LLVM_BUILD_LLVM_C_DYLIB", "CLANG_LINK_CLANG_DYLIB", "CLANG_SPAWN_CC1", "LLVM_ENABLE_PLUGINS"]) {assert.equal(d[name], "OFF", name);}
    for (const name of ["SYCL_JIT_STATIC", "LLVM_ENABLE_EH", "LLVM_ENABLE_RTTI"]) {assert.equal(d[name], "ON", name);}
  });
  test("explicitly selects immutable resource and compiler inputs", () => {
    const d = plan().definitions;
    assert.equal(d["SYCL_JIT_RESOURCE_OVERLAY"], "/resources");
    assert.equal(d["CMAKE_C_COMPILER"], "/compiler/bin/clang"); assert.equal(d["CMAKE_CXX_COMPILER"], "/compiler/bin/clang++");
  });
  test("omits development-only examples, benchmarks, tests and optional shared dependencies", () => {
    const d = plan().definitions;
    for (const name of ["LLVM_INCLUDE_EXAMPLES", "LLVM_INCLUDE_BENCHMARKS", "LLVM_INCLUDE_TESTS", "CLANG_BUILD_EXAMPLES", "CLANG_INCLUDE_TESTS", "LLVM_ENABLE_LIBXML2", "LLVM_ENABLE_ZLIB", "LLVM_ENABLE_ZSTD"]) {assert.equal(d[name], "OFF", name);}
  });
  test("builds the upstream transitive closure while collecting independent failures", () => {
    assert.deepEqual(plan().compile, [tools.cmake, "--build", "/output/build", "--target", "native-dpcpp-jit", "--parallel", "2", "--", "-k", "0"]);
  });
  test("Windows requires an explicit cross-toolchain rather than inheriting host discovery", () => {
    assert.throws(() => jitSdkPlan("windows", roots, "/compiler", "/resources", "/output", tools, 2), /Static JIT SDK/);
    const p = jitSdkPlan("windows", roots, "/compiler", "/resources", "/output", {...tools, toolchain: "/sdk/toolchain.cmake"}, 2);
    assert.equal(p.definitions["CMAKE_CXX_COMPILER"], "/compiler/bin/clang-cl"); assert.equal(p.definitions["CMAKE_TOOLCHAIN_FILE"], "/sdk/toolchain.cmake");
  });
  test("Linux can select a compatible sysroot through an explicit toolchain", () => {
    assert.equal(jitSdkPlan("linux", roots, "/compiler", "/resources", "/output", {...tools, toolchain: "/sdk/linux.cmake"}, 2).definitions["CMAKE_TOOLCHAIN_FILE"], "/sdk/linux.cmake");
  });
  test("literal paths with spaces remain single arguments rather than shell flags", () => {
    const p = jitSdkPlan("linux", roots, "/compiler space", "/resource space", "/output space", tools, 2);
    assert.ok(p.configure.includes("-DCMAKE_CXX_COMPILER=/compiler space/bin/clang++"));
    assert.ok(p.configure.includes("-DSYCL_JIT_RESOURCE_OVERLAY=/resource space"));
  });
  test("rejects missing or invented source components", () => {
    const {spirvHeaders: _, ...missing} = roots;
    for (const selected of [missing, {...roots, invented: "/inputs/invented"}]) {assert.throws(() => jitSdkPlan("linux", selected, "/compiler", "/resources", "/output", tools, 2));}
  });
  test("rejects nonliteral paths before any tool execution or output mutation", () => {
    for (const sdk of ["relative", "/a;bad", "/a\nbad", "/a$bad", "/a/../b", '/a"bad', "/a\\bad"]) {assert.throws(() => jitSdkPlan("linux", roots, sdk, "/resources", "/output", tools, 2));}
  });
  test("rejects ineffective or excessive parallelism", () => {
    for (const jobs of [0, 1.5, 17, NaN, Infinity]) {assert.throws(() => jitSdkPlan("linux", roots, "/compiler", "/resources", "/output", tools, jobs));}
  });
  test("rejects unrelated build settings and unsupported target aliases", () => {
    const extra = {...tools, command: "/shell"};
    assert.throws(() => jitSdkPlan("linux", roots, "/compiler", "/resources", "/output", extra, 2));
    for (const target of ["", "win32", "all"]) {assert.throws(() => Reflect.apply(jitSdkPlan, null, [target, roots, "/compiler", "/resources", "/output", tools, 2]));}
  });
  test("output cannot enter, contain or replace source, resource or tool inputs", () => {
    for (const output of ["/inputs", "/inputs/dpcpp/output", "/resources/out", "/compiler", "/tools", root]) {assert.throws(() => jitSdkPlan("linux", roots, "/compiler", "/resources", output, tools, 2), /overlaps/);}
  });
  test("returned plans do not mutate later plans or source policy", () => {
    const before = sdkComponents("compiler"), p = plan(); p.definitions["LLVM_TARGETS_TO_BUILD"] = "bad"; p.compile.length = 0;
    assert.equal(plan().definitions["LLVM_TARGETS_TO_BUILD"], "X86;NVPTX;SPIRV"); assert.deepEqual(sdkComponents("compiler"), before);
  });
});

test.describe("Windows original compiler target archive discovery", () => {
  const archives = ["lib/sycl-jit.lib", ...Array.from({length: 50}, (_, index) => `lib/LLVM${index}.lib`)];
  const link = "cmake -E vs_link_exe -- lld-link /out:bin/native-dpcpp-jit.exe /implib:lib/native-dpcpp-jit.lib " + archives.join(" ") + " kernel32.lib libcmt.lib";
  test("uses original ordered compiler targets without a maintained Windows library list", () => {
    assert.deepEqual(compilerArchives(link, "/original", true), archives.map(name => path.join("/original", name)));
  });
  test("ignores unrelated archive creation commands", () => {
    assert.equal(compilerArchives("llvm-lib /out:lib/Other.lib other.obj\n" + link, "/original", true).length, 51);
  });
  test("ordinary OS and static CRT bare libraries are not compiler targets", () => {
    assert.ok(compilerArchives(link + " advapi32.lib libucrt.lib", "/original", true).every(file => path.dirname(file) === path.join("/original", "lib")));
  });
  for (const [name, changed] of [
    ["duplicate target links", link + "\n" + link],
    ["duplicate archive targets", link + " lib/LLVM0.lib"],
    ["absolute compiler archive paths", link.replace("lib/LLVM0.lib", "/external/LLVM0.lib")],
    ["traversal archive paths", link.replace("lib/LLVM0.lib", "lib/../escape.lib")],
    ["Windows path separators outside the actual CMake grammar", link.replace("lib/LLVM0.lib", "lib\\LLVM0.lib")],
    ["mixed ELF archive targets", link.replace("lib/LLVM0.lib", "lib/LLVM0.a")],
    ["archive-looking shell tokens", link.replace("lib/LLVM0.lib", "lib/LLVM0.lib;bad")],
    ["a truncated compiler closure", link.replace(archives.slice(2).join(" "), "")],
    ["wrong original entry archive", link.replace("lib/sycl-jit.lib", "lib/another.lib")],
    ["wrong executable target", link.replace("/out:bin/native-dpcpp-jit.exe", "/out:bin/other.exe")]
  ]) {
    test("rejects " + name + " without executing the generated command", () => {
      assert.throws(() => compilerArchives(changed || "", "/original", true), /Unrecognized/);
    });
  }
});

/**
 * Configure the real fixture in a disposable project. The imported compiler target is only a
 * stand-in for the original build graph; no compiler or JIT code is built by this test.
 * @param {import("node:test").TestContext} context
 * @param {boolean} windows
 * @param {string} [compilerId]
 * @param {boolean} [pchFixture]
 * @returns {{generated: string, build: string}}
 */
function configureFixture(context, windows, compilerId = "Clang", pchFixture = false) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-cmake-"));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const source = path.join(directory, "CMakeLists.txt"), build = path.join(directory, "build");
  const cmakeDollar = String.fromCharCode(36);
  const cmakeBinaryDir = `${cmakeDollar}{CMAKE_BINARY_DIR}`, cmakeRepo = `${cmakeDollar}{MOM_SOURCE_REPO}`;
  const sourceRoot = directory.replace(/\\/g, "/"), lines = [
    "cmake_minimum_required(VERSION 3.20)",
    "project(mom_windows_jit_fixture LANGUAGES CXX)",
    `file(WRITE "${cmakeBinaryDir}/sycl-jit.lib" "")`,
    "add_library(sycl-jit STATIC IMPORTED GLOBAL)",
    `set_property(TARGET sycl-jit PROPERTY IMPORTED_LOCATION "${cmakeBinaryDir}/sycl-jit.lib")`,
  ];
  if (pchFixture) {
    lines.push(
      "add_library(LLVMAnalysis STATIC lib/Analysis/HeatUtils.cpp lib/Analysis/Other.cpp)",
      `target_precompile_headers(LLVMAnalysis PRIVATE "${sourceRoot}/lib/Analysis/pch.hpp")`,
    );
  }
  lines.push(
    "set(SYCL_JIT_STATIC ON)",
    `set(MOM_SOURCE_REPO "${root.replace(/\\/g, "/")}")`,
    `set(LLVM_EXTERNAL_SYCL_SOURCE_DIR "${cmakeRepo}")`,
    `set(SYCL_JIT_RESOURCE_OVERLAY "${cmakeRepo}")`,
    `include("${path.join(root, "tests/native/dpcpp-jit/CMakeLists.txt").replace(/\\/g, "/")}")`,
  );
  if (pchFixture) {
    lines.push(
      `get_source_file_property(heatutils_skip "${sourceRoot}/lib/Analysis/HeatUtils.cpp" TARGET_DIRECTORY LLVMAnalysis SKIP_PRECOMPILE_HEADERS)`,
      `get_source_file_property(other_skip "${sourceRoot}/lib/Analysis/Other.cpp" TARGET_DIRECTORY LLVMAnalysis SKIP_PRECOMPILE_HEADERS)`,
      `file(WRITE "${cmakeBinaryDir}/pch-state.txt" "\${heatutils_skip}\\n\${other_skip}\\n\${WIN32}\\n\${CMAKE_CXX_COMPILER_ID}\\n")`,
    );
  }
  lines.push("");
  if (pchFixture) {
    const analysis = path.join(directory, "lib", "Analysis");
    fs.mkdirSync(analysis, {recursive: true});
    fs.writeFileSync(path.join(analysis, "HeatUtils.cpp"), "#include \"pch.hpp\"\nint heatutils_pch_probe() { return pch_probe_value; }\n");
    fs.writeFileSync(path.join(analysis, "Other.cpp"), "#include \"pch.hpp\"\nint other_pch_probe() { return pch_probe_value; }\n");
    fs.writeFileSync(path.join(analysis, "pch.hpp"), "#pragma once\ninline constexpr int pch_probe_value = 7;\n");
  }
  fs.writeFileSync(source, lines.join("\n"));
  // This is deliberately a configure-only contract test. Use the running Node executable as a
  // harmless compiler descriptor and force the identity for both OS branches; no generated
  // target is compiled or used to qualify a toolchain. Setting the system name explicitly keeps
  // the Linux branch deterministic when the suite itself runs on a Windows host.
  const args = ["-S", directory, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"];
  const systemName = windows ? "Windows" : "Linux";
  const cmakeCompiler = process.execPath.replace(/\\/g, "/");
  args.push(`-DCMAKE_SYSTEM_NAME=${systemName}`, `-DCMAKE_CXX_COMPILER=${cmakeCompiler}`,
    "-DCMAKE_CXX_COMPILER_WORKS=TRUE", "-DCMAKE_CXX_COMPILER_FORCED=TRUE",
    `-DCMAKE_CXX_COMPILER_ID=${compilerId}`, "-DCMAKE_CXX_COMPILER_VERSION=23.0.0",
    "-DCMAKE_CXX_COMPILE_FEATURES=cxx_std_17",
    `-DCMAKE_CXX_COMPILER_FRONTEND_VARIANT=${windows ? "MSVC" : "GNU"}`, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
  if (windows) {
    args.push("-DCMAKE_CXX_SIMULATE_ID=MSVC", `-DCMAKE_RC_COMPILER=${cmakeCompiler}`,
      "-DCMAKE_RC_COMPILER_WORKS=TRUE", "-DCMAKE_RC_COMPILER_FORCED=TRUE");
  }
  const result = spawnSync("cmake", args, {encoding: "utf8", timeout: 30000});
  assert.equal(result.status, 0, `${windows ? "Windows" : "Linux"} CMake configure failed:\n${result.stdout}\n${result.stderr}`);
  const generated = fs.readFileSync(path.join(build, "build.ninja"), "utf8");
  assert.match(generated, /native-dpcpp-jit/);
  return {generated, build};
}

/** @param {string} file @returns {string} */
function patchSection(file) {
  const marker = `diff --git a/${file} b/${file}`, start = staticJitPatch.indexOf(marker);
  if (start < 0) {return "";}
  const end = staticJitPatch.indexOf("\ndiff --git ", start + marker.length);
  return staticJitPatch.slice(start, end < 0 ? undefined : end);
}

/** @param {string} section @returns {string} */
function addedPatchLines(section) {
  return section.split("\n").filter(line => line.startsWith("+") && !line.startsWith("+++"))
    .map(line => line.slice(1)).join("\n");
}

/** @param {string} patch */
function assertPatchHunkCounts(patch) {
  const lines = patch.split("\n");
  if (lines.at(-1) === "") {lines.pop();}
  for (let index = 0; index < lines.length; index++) {
    const match = lines[index]?.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
    if (!match) {continue;}
    let oldLines = 0, newLines = 0, cursor = index + 1;
    for (; cursor < lines.length && !/^(?:@@ |diff --git )/.test(lines[cursor] || ""); cursor++) {
      const line = lines[cursor] || "";
      if (line.startsWith(" ")) {
        oldLines++; newLines++;
      } else if (line.startsWith("-")) {
        oldLines++;
      } else if (line.startsWith("+")) {
        newLines++;
      } else {assert.fail(`Malformed unified-diff line ${cursor + 1}: ${JSON.stringify(line)}`);}
    }
    assert.equal(oldLines, Number(match[2] || 1), `wrong old-line count at hunk ${index + 1}`);
    assert.equal(newLines, Number(match[4] || 1), `wrong new-line count at hunk ${index + 1}`);
    index = cursor - 1;
  }
}

const cmakePatch = patchSection("sycl-jit/jit-compiler/CMakeLists.txt");
const deviceCompilationPatch = patchSection("sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp");
const deviceCompilationAdded = addedPatchLines(deviceCompilationPatch);
const windowsDeviceProfile = deviceCompilationAdded.match(/#ifdef MOM_DPCPP_STATIC_JIT\n[ ]{2}#ifdef _WIN32[\s\S]*?[ ]{2}#endif\n[ ]{4}DAL\.AddFlagArg\(nullptr, OptTable\.getOption\(OPT_nostdinc\)\);/)?.[0] || "";
const cmakeAddedLines = addedPatchLines(cmakePatch).split("\n");
const resourceClangStart = cmakeAddedLines.indexOf("  if(SYCL_JIT_RESOURCE_CLANG)");
const resourceClangEnd = resourceClangStart < 0 ? -1 : cmakeAddedLines.indexOf("  endif()", resourceClangStart);
const resourceClangSelection = resourceClangStart < 0 || resourceClangEnd < 0 ?
  "" : cmakeAddedLines.slice(resourceClangStart, resourceClangEnd + 1).join("\n");

test.describe("Static source-JIT optional compiler tools", () => {
  /** Real CMake target validation, without compiling LLVM, a module, or a mining kernel.
   * @param {import('node:test').TestContext} t @param {string} body */
  function configure(t, body) {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-tools-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    fs.writeFileSync(path.join(directory, "empty.cpp"), "int mom_jit_fixture;\n");
    fs.writeFileSync(path.join(directory, "CMakeLists.txt"), "cmake_minimum_required(VERSION 3.20)\nproject(jit_tools CXX)\n" + body);
    return spawnSync("cmake", ["-S", directory, "-B", path.join(directory, "build"), "-G", "Ninja"], {encoding: "utf8", timeout: 30000});
  }
  for (const windows of [false, true]) {
    for (const plugins of [false, true]) {
      test(`${windows ? "Windows" : "Linux"} keeps the static passes when plugins are ${plugins ? "enabled" : "disabled"}`, t => {
        const branch = addedPatchLines(patchSection("sycl-jit/passes/CMakeLists.txt")).match(/^if\(.+\)$/m)?.[0];
        assert.ok(branch, "The plugin must respect LLVM's target-availability gate");
        const body = `set(WIN32 ${windows ? "TRUE" : "FALSE"})\nset(LLVM_ENABLE_PLUGINS ${plugins ? "ON" : "OFF"})\n` +
          "function(add_llvm_library name)\nif(LLVM_ENABLE_PLUGINS)\nadd_library(@{name} MODULE empty.cpp)\nelse()\nadd_custom_target(@{name})\nendif()\nendfunction()\n".replaceAll("@", String.fromCharCode(36)) +
          branch + "\nadd_llvm_library(SYCLJITPassesPlugin)\ntarget_compile_options(SYCLJITPassesPlugin PRIVATE -Wall)\nendif()\n" +
          "add_library(SYCLJITPasses STATIC empty.cpp)\nget_target_property(kind SYCLJITPasses TYPE)\nif(NOT kind STREQUAL STATIC_LIBRARY)\nmessage(FATAL_ERROR \"Missing static JIT passes\")\nendif()\n";
        const result = configure(t, body); assert.equal(result.status, 0, result.stderr);
        if (!windows && !plugins) {
          const before = configure(t, body.replace(branch, "if(NOT WIN32 AND NOT CYGWIN)"));
          assert.notEqual(before.status, 0); assert.match(before.stderr, /non-compilable target type/);
        }
      });
    }
  }
  for (const tests of [false, true]) {
    test(`source-JIT test tools are ${tests ? "requested explicitly" : "omitted from the release compiler graph"}`, t => {
      const branch = addedPatchLines(patchSection("sycl-jit/CMakeLists.txt")).match(/^if\(NOT WIN32 AND NOT CYGWIN AND LLVM_INCLUDE_TESTS\)$/m)?.[0];
      assert.ok(branch, "The original test directory must respect LLVM_INCLUDE_TESTS");
      const body = `set(WIN32 FALSE)\nset(CYGWIN FALSE)\nset(LLVM_INCLUDE_TESTS ${tests ? "ON" : "OFF"})\n${branch}\nset(mom_test_tools ON)\nendif()\n` +
        `if(NOT "\${mom_test_tools}" STREQUAL "${tests ? "ON" : ""}")\nmessage(FATAL_ERROR "Wrong test-tool selection")\nendif()\n`;
      const result = configure(t, body); assert.equal(result.status, 0, result.stderr);
    });
  }
});

test.describe("Static source-JIT immutable resource selection", () => {
  for (const native of [true, false]) {
    test(native ? "static mode neither installs SDKs nor merges stale generated headers" : "shared mode preserves its original SDK install dependencies and outputs", t => {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-overlay-cmake-"));
      t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
      const start = cmakeAddedLines.findIndex(line => /^set\(SYCL_JIT_RESOURCE_BYPRODUCTS /.test(line));
      const end = cmakeAddedLines.indexOf("endif()", start); assert.ok(start >= 0 && end > start);
      const body = cmakeAddedLines.slice(start, end + 1).join("\n"), source = path.join(directory, "CMakeLists.txt");
      const overlay = path.join(directory, "overlay").replace(/\\/g, "/"); fs.mkdirSync(overlay);
      fs.writeFileSync(path.join(overlay, "correct-header.h"), "immutable input\n");
      const original = path.join(directory, "sdk-output").replace(/\\/g, "/");
      fs.writeFileSync(source, "cmake_minimum_required(VERSION 3.20)\nproject(overlay NONE)\n" +
        `set(SYCL_JIT_STATIC ${native ? "ON" : "OFF"})\nset(SYCL_JIT_RESOURCE_OVERLAY "${overlay}")\n` +
        `set(SYCL_JIT_RESOURCE_INSTALL_DIR "${original}")\n` +
        "set(SYCL_JIT_RESOURCE_FILES stale-linux-header.h)\nset(SYCL_JIT_RESOURCE_DEPS shared-sdk-tool)\n" +
        "set(SYCL_JIT_PREPARE_RESOURCE_COMMANDS original-install-command)\n" + body +
        "\nfile(WRITE \"@{CMAKE_BINARY_DIR}/result.txt\" \"@{SYCL_JIT_RESOURCE_INSTALL_DIR}\\n@{SYCL_JIT_RESOURCE_FILES}\\n@{SYCL_JIT_RESOURCE_DEPS}\\n@{SYCL_JIT_PREPARE_RESOURCE_COMMANDS}\\n@{SYCL_JIT_RESOURCE_BYPRODUCTS}\\n\")\n".replaceAll("@", String.fromCharCode(36)));
      const build = path.join(directory, "build"), result = spawnSync("cmake", ["-S", directory, "-B", build, "-G", "Ninja"], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.stderr);
      assert.deepEqual(fs.readFileSync(path.join(build, "result.txt"), "utf8").trimEnd().split(/\r?\n/), native ?
        [overlay, overlay + "/correct-header.h"] : [original, "stale-linux-header.h", "shared-sdk-tool", "original-install-command", "stale-linux-header.h"]);
    });
  }
  test("hermetic Windows compilation keeps explicitly admitted LLVM include roots", t => {
    const lookup = spawnSync(process.platform === "win32" ? "where.exe" : "which", ["clang-cl"], {encoding: "utf8", timeout: 5000});
    const clang = process.env["MOM_TEST_LLVM_BIN"] ? path.join(process.env["MOM_TEST_LLVM_BIN"], "clang-cl" + (process.platform === "win32" ? ".exe" : "")) : lookup.stdout?.trim().split(/\r?\n/)[0];
    if (!clang || !fs.statSync(clang, {throwIfNoEntry: false})?.isFile()) {t.skip("headerless clang-cl unavailable"); return;}
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-hermetic-includes-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    fs.writeFileSync(path.join(directory, "owned.h"), "#define MOM_OWNED_HEADER 17\n");
    const source = path.join(directory, "probe.cpp"), output = path.join(directory, "probe.obj");
    fs.writeFileSync(source, "#include <owned.h>\nint value = MOM_OWNED_HEADER;\n");
    const branch = addedPatchLines(patchSection("sycl-jit/CMakeLists.txt"));
    assert.match(branch, /if\(WIN32 AND SYCL_JIT_STATIC\)[\s\S]*set\(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-I"\)/);
    const before = spawnSync(clang, ["--target=x86_64-pc-windows-msvc", "/nologo", "-nostdinc", "-imsvc" + directory, "/c", source, "/Fo" + output], {encoding: "utf8", timeout: 30000});
    assert.notEqual(before.status, 0); assert.match(before.stderr, /owned\.h.*file not found/);
    const after = spawnSync(clang, ["--target=x86_64-pc-windows-msvc", "/nologo", "-nostdinc", "-I" + directory, "/c", source, "/Fo" + output], {encoding: "utf8", timeout: 30000});
    assert.equal(after.status, 0, after.stderr); assert.ok(fs.statSync(output).size > 0);
  });
});

/** @param {string} file @returns {string} */
function cmakePath(file) {
  return "\\$\\{MOM_SOURCE_REPO\\}/" + file.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** @param {string[]} files @returns {string} */
function manifestHash(files) {
  const hash = createHash("sha256");
  for (const file of files) {
    const absolute = path.join(root, file), stat = fs.statSync(absolute);
    assert.ok(stat.isFile(), `Source manifest entry is not a regular file: ${file}`);
    hash.update(file).update("\0").update(fs.readFileSync(absolute));
  }
  return hash.digest("hex");
}

const windowsStart = cmake.indexOf("if(WIN32)\n  # Keep the source-JIT fixture");
const windowsEnd = cmake.indexOf("else()", windowsStart);
const windowsBranch = cmake.slice(windowsStart, windowsEnd);
const linuxBranch = cmake.slice(windowsEnd, cmake.indexOf("endif()", windowsEnd));
const resourceCommand = cmake.match(/add_custom_command\(OUTPUT[\s\S]*?\n\s+VERBATIM\)/)?.[0] || "";

test.describe("Windows pinned static source-JIT build contract", () => {
  test("static PTX avoids installation discovery while retaining embedded libdevice linking", () => {
    assert.match(deviceCompilationAdded, /if \(Format == BinaryFormat::PTX\) \{[\s\S]*?DAL\.AddFlagArg\(nullptr, OptTable\.getOption\(OPT_no_offloadlib\)\);/);
    assert.match(deviceCompilationAdded, /LibDeviceFiles\.push_back\(\(TC\.getPrefix\(\) \+ "\/cuda\/nvvm\/libdevice\/libdevice\.10\.bc"\)\.str\(\)\);/);
    assert.doesNotMatch(deviceCompilationAdded, /OPT_cuda_path_EQ/);
  });

  test("static PTX forwards the selected PTX feature rather than inferring it from a CUDA SDK", () => {
    assert.match(deviceCompilationAdded, /StringRef\(Features\)\.split\(TargetFeatures, ','\);/);
    assert.match(deviceCompilationAdded, /if \(Feature\.starts_with\("\+ptx"\)\)\s+DAL\.AddJoinedArg\(nullptr, OptTable\.getOption\(OPT_cuda_feature_EQ\), Feature\);/);
  });

  test("keeps the GNU device profile inside static JIT only", () => {
    assert.ok(deviceCompilationPatch, "The maintained patch must include DeviceCompilation.cpp");
    assert.match(windowsDeviceProfile, /#ifdef MOM_DPCPP_STATIC_JIT\n[ ]{2}#ifdef _WIN32/);
    assert.match(windowsDeviceProfile, /DAL\.AddJoinedArg\(nullptr, OptTable\.getOption\(OPT_target\),\n[ ]{21}"x86_64-w64-windows-gnu"\);/);
    assert.deepEqual([...windowsDeviceProfile.matchAll(/"(__SSE__|__SSE2__|__MMX__|__AVX__)"/g)].map(match => match[1]),
      ["__SSE__", "__SSE2__", "__MMX__", "__AVX__"]);
    assert.equal((windowsDeviceProfile.match(/OPT_U/g) || []).length, 1);
    const outside = deviceCompilationAdded.replace(windowsDeviceProfile, "");
    assert.doesNotMatch(outside, /x86_64-w64-windows-gnu|OPT_target|OPT_U/);
  });

  test("keeps the resource host-compiler override Windows-only and opt-in", () => {
    assert.match(cmakePatch, /set\(SYCL_JIT_RESOURCE_CLANG "" CACHE FILEPATH "Host clang-cl used to compile embedded JIT resources"\)/);
    assert.match(resourceClangSelection, /^[ ]{2}if\(SYCL_JIT_RESOURCE_CLANG\)[\s\S]*^[ ]{2}endif\(\)$/m);
    assert.match(resourceClangSelection, /set\(clang_exe "\$\{SYCL_JIT_RESOURCE_CLANG\}"\)/);
    assert.match(resourceClangSelection, /set\(clang_exe \$\{CMAKE_BINARY_DIR\}\/bin\/clang-cl\.exe\)/);
  });

  test("retains valid unified-diff hunk counts", () => {assertPatchHunkCounts(staticJitPatch);});

  test("applies the patch to the pinned clean DPC++ source when supplied", (t) => {
    const source = process.env["MOM_DPCPP_SOURCE"];
    if (!source) {t.skip("set MOM_DPCPP_SOURCE to the pinned clean DPC++ checkout"); return;}
    assert.equal(path.isAbsolute(source), true, "MOM_DPCPP_SOURCE must be absolute");
    const revision = spawnSync("git", ["-C", source, "rev-parse", "HEAD"], {encoding: "utf8", timeout: 10000});
    assert.equal(revision.status, 0, revision.stderr || "cannot inspect MOM_DPCPP_SOURCE");
    assert.equal(revision.stdout.trim(), "190878e023dd0979199786b19604b377229a4996");
    const status = spawnSync("git", ["-C", source, "status", "--porcelain=v1", "--untracked-files=all"], {encoding: "utf8", timeout: 10000});
    assert.equal(status.status, 0, status.stderr || "cannot inspect MOM_DPCPP_SOURCE status");
    assert.equal(status.stdout, "", "MOM_DPCPP_SOURCE must be clean and have no untracked files");
    const check = spawnSync("git", ["-C", source, "apply", "--check", "--whitespace=error", path.join(root, "runtime/dpcpp-static-jit.patch")], {encoding: "utf8", timeout: 30000});
    assert.equal(check.status, 0, check.stderr || check.stdout || "static JIT patch does not apply");
  });

  test("attaches to the original static compiler target only", () => {
    assert.match(cmake, /TARGET sycl-jit OR NOT SYCL_JIT_STATIC/);
    assert.match(cmake, /get_target_property\(jit_type sycl-jit TYPE\)/);
    assert.match(cmake, /jit_type STREQUAL "STATIC_LIBRARY"/);
    assert.match(cmake, /target_link_libraries\(native-dpcpp-jit PRIVATE sycl-jit/);
    assert.match(cmake, /if\(NOT WIN32\)[\s\S]*list\(APPEND native_dpcpp_jit_sources[\s\S]*runtime\/driver-library\.cpp"\)/);
    assert.doesNotMatch(cmake, /add_library\(sycl-jit|MOM_DPCPP_JIT_OBJECT/);
  });

  test("keeps the maintained source and resource manifest explicit and fresh", () => {
    assert.equal(sourceManifest.length, new Set(sourceManifest).size);
    assert.match(manifestHash(sourceManifest), /^[a-f0-9]{64}$/);
    for (const source of sourceManifest.slice(1, 4)) {assert.match(cmake, new RegExp(cmakePath(source)));}
    for (const input of ["scripts/native-resources.js", "sycl/kawpow/device.inc", "sycl/kawpow/keccak.inc"]) {
      assert.match(resourceCommand, new RegExp(cmakePath(input)));
    }
    assert.match(resourceCommand, /DEPENDS[\s\S]*native-resources\.js[\s\S]*device\.inc[\s\S]*keccak\.inc/);
    assert.match(cmake, /target_sources\(native-dpcpp-jit PRIVATE "\$\{jit_resources\}"\)/);
  });

  test("uses a virtual source name valid on both platform path conventions", () => {
    const name = nativeJitSource.match(/const jit::InMemoryFile\s+file\s*\{\s*"([^"]+)"/)?.[1];
    assert.ok(name, "The source fixture must name its virtual input");
    assert.equal(path.posix.isAbsolute(name), false);
    assert.equal(path.win32.isAbsolute(name), false);
    assert.doesNotMatch(name, /^[A-Za-z]:/);
  });

  test("emits maintained KawPow source before any JIT or driver operation", (t) => {
    const branch = nativeJitSource.indexOf('std::strcmp(argv[1], "--emit-source")'), configure = nativeJitSource.indexOf("configure(", branch);
    assert.ok(branch > 0 && configure > branch, "The build-only source branch must precede JIT configuration");
    assert.match(nativeJitSource.slice(0, configure), /require\(argc == 3, "Expected --emit-source <algorithm>"\)/);
    assert.match(nativeJitSource.slice(0, configure), /nativeEmitKawpowSource\(argv\[2\]\)/);
    assert.match(nativeSourceEmitter, /_setmode\(_fileno\(stdout\), _O_BINARY\)/);
    assert.match(nativeSourceEmitter, /std::fwrite\(generated\.second\.data\(\)/);
    assert.match(nativeJitSource.slice(0, configure), /return 0;/);
    const fixture = process.env["MOM_NATIVE_DPCPP_JIT_FIXTURE"];
    if (!fixture) {t.skip("set MOM_NATIVE_DPCPP_JIT_FIXTURE for the optional built-fixture emission check"); return;}
    assert.equal(path.isAbsolute(fixture), true, "MOM_NATIVE_DPCPP_JIT_FIXTURE must be absolute");
    /** @type {Map<string, Buffer>} */
    const outputs = new Map();
    for (const algorithm of ["kawpow", "firopow", "evrprogpow", "meowpow"]) {
      const result = /** @type {import("node:child_process").SpawnSyncReturns<Buffer>} */
        (spawnSync(fixture, ["--emit-source", algorithm], {encoding: null, timeout: 120000, maxBuffer: 16 << 20}));
      assert.equal(result.status, 0, result.stderr?.toString() || "source emission failed");
      assert.ok(Buffer.isBuffer(result.stdout) && result.stdout.length > 0, "source emission must return bytes");
      outputs.set(algorithm, /** @type {Buffer} */ (result.stdout));
    }
    for (const [algorithm, bytes] of outputs) {
      const again = /** @type {import("node:child_process").SpawnSyncReturns<Buffer>} */
        (spawnSync(fixture, ["--emit-source", algorithm], {encoding: null, timeout: 120000, maxBuffer: 16 << 20}));
      assert.equal(again.status, 0, again.stderr?.toString() || "source determinism failed");
      assert.deepEqual(again.stdout, bytes, `${algorithm} source changed between emissions`);
    }
  });

  test("uses one static CRT and only the original Windows system closure", () => {
    assert.match(windowsBranch, /MSVC_RUNTIME_LIBRARY MultiThreaded/);
    assert.match(windowsBranch, /target_compile_options\(native-dpcpp-jit PRIVATE \/EHsc\)/);
    assert.match(windowsBranch, /target_link_options\(native-dpcpp-jit PRIVATE \/SUBSYSTEM:CONSOLE \/INCREMENTAL:NO\)/);
    for (const library of windowsSystemLibraries) {assert.match(windowsBranch, new RegExp(`\\b${library}\\b`));}
    assert.doesNotMatch(windowsBranch, /\/MD|LLVM[^\s)]*\.lib|clang[^\s)]*\.lib/);
  });

  test("leaves the existing Linux static link policy in its non-Windows branch", () => {
    for (const flag of linuxLinkFlags) {assert.match(linuxBranch, new RegExp(flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));}
    assert.match(linuxBranch, /target_link_libraries\(native-dpcpp-jit PRIVATE sycl-jit dl pthread m\)/);
    assert.doesNotMatch(linuxBranch, /MSVC_RUNTIME_LIBRARY|\/SUBSYSTEM|\/INCREMENTAL/);
  });

  test("does not add a compiler registry, DLL loader, or extracted build input", () => {
    assert.doesNotMatch(cmake, /target_link_libraries\([^)]*\b(?:LLVM|clang)[^)]*\)/i);
    assert.doesNotMatch(cmake, /LoadLibrary|GetProcAddress|find_library|add_library\([^)]*SHARED/i);
    assert.match(cmake, /foreach\(system_dll IN ITEMS shell32 ole32\)/);
    assert.equal((cmake.match(/\.dll/gi) || []).length, 2, "only the two admitted OS delay-load names may mention DLLs");
    assert.doesNotMatch(cmake, /file\(GLOB|execute_process|extract|ExternalProject|FetchContent/);
  });

  test("configures the maintained fixture on Linux and Windows", (t) => {
    for (const tool of ["cmake", "ninja"]) {
      const probe = spawnSync(tool, ["--version"], {encoding: "utf8", timeout: 5000});
      if (probe.status !== 0 || probe.error) {t.skip(`${tool} unavailable`); return;}
    }
    const {generated: linux} = configureFixture(t, false), {generated: windows} = configureFixture(t, true);
    for (const flag of linuxLinkFlags) {assert.match(linux, new RegExp(flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));}
    for (const library of windowsSystemLibraries) {assert.match(windows, new RegExp(library));}
    assert.match(windows, /SUBSYSTEM:CONSOLE/);
    assert.match(windows, /INCREMENTAL:NO/);
    assert.match(windows, /\/EHsc/);
    assert.match(windows, /INCLUDES = -I/);
    assert.doesNotMatch(windows, /-imsvc/);
  });

  test("Windows IntelLLVM skips only HeatUtils from LLVMAnalysis PCH use", t => {
    for (const tool of ["cmake", "ninja"]) {
      const probe = spawnSync(tool, ["--version"], {encoding: "utf8", timeout: 5000});
      if (probe.status !== 0 || probe.error) {t.skip(`${tool} unavailable`); return;}
    }
    const fixture = configureFixture(t, true, "IntelLLVM", true);
    assert.deepEqual(fs.readFileSync(path.join(fixture.build, "pch-state.txt"), "utf8").trimEnd().split(/\r?\n/), ["ON", "NOTFOUND", "1", "IntelLLVM"]);
    const commands = /** @type {Array<{file: string, command: string}>} */ (JSON.parse(fs.readFileSync(path.join(fixture.build, "compile_commands.json"), "utf8")));
    const heat = commands.find(row => path.basename(row.file) === "HeatUtils.cpp");
    const other = commands.find(row => path.basename(row.file) === "Other.cpp");
    assert.ok(heat); assert.ok(other);
    /** @param {{command: string}} row @returns {boolean} */
    const consumesPch = row => /(?:\/Y[uc]\S*|\/Fp\S*|\/FI\S*|-include-pch(?:\s|$)|(?:^|\s)-include\s+[^\s]*cmake_pch)/i.test(row.command);
    assert.equal(consumesPch(heat), false, "HeatUtils must not consume LLVMAnalysis's target PCH");
    assert.equal(consumesPch(other), true, "unrelated LLVMAnalysis sources must retain target PCH use");

    const clangWindows = configureFixture(t, true, "Clang", true);
    const intelLinux = configureFixture(t, false, "IntelLLVM", true);
    assert.deepEqual(fs.readFileSync(path.join(clangWindows.build, "pch-state.txt"), "utf8").trimEnd().split(/\r?\n/), ["NOTFOUND", "NOTFOUND", "1", "Clang"]);
    assert.deepEqual(fs.readFileSync(path.join(intelLinux.build, "pch-state.txt"), "utf8").trimEnd().split(/\r?\n/), ["NOTFOUND", "NOTFOUND", "", "IntelLLVM"]);
  });

  test("executes the Windows host compiler selection with cmake -P", (t) => {
    const probe = spawnSync("cmake", ["--version"], {encoding: "utf8", timeout: 5000});
    if (probe.status !== 0 || probe.error) {t.skip("cmake unavailable"); return;}
    assert.ok(resourceClangSelection, "The patch must contain the host compiler selection fragment");
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-resource-profile-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const script = path.join(directory, "profile.cmake"), result = path.join(directory, "selected.txt");
    /** @param {string} value */
    const normalize = value => value.replace(/\\/g, "/");
    /** @param {string} resourceClang */
    function selectCompiler(resourceClang) {
      fs.writeFileSync(script, [
        "cmake_minimum_required(VERSION 3.20)",
        `set(CMAKE_BINARY_DIR "${normalize(directory)}")`,
        `set(SYCL_JIT_RESOURCE_CLANG "${normalize(resourceClang)}")`,
        resourceClangSelection,
        `file(WRITE "${normalize(result)}" "\${clang_exe}\\n")`,
        "",
      ].join("\n"));
      const run = spawnSync("cmake", ["-P", script], {encoding: "utf8", timeout: 10000});
      assert.equal(run.status, 0, run.error?.message || `${run.stdout}\n${run.stderr}`);
      return fs.readFileSync(result, "utf8").trim();
    }
    assert.equal(selectCompiler(""), normalize(path.join(directory, "bin", "clang-cl.exe")));
    const selected = normalize(path.join(directory, "host", "clang-cl"));
    assert.equal(selectCompiler(selected), selected);
  });
});
