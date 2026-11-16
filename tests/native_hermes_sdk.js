"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {hermesPins, hermesSdkPlan, hermesSourceFiles, validateHermesResumeCache, copyHermesSourceChanges,
  hermesResumeBaseline, buildNativeHermes} = require("../scripts/build-native-hermes");
const {nativeSdkEnvironment} = require("../scripts/build-environment");
const {fileHash} = require("../scripts/build-source-inputs");
const {spawnSync} = require("node:child_process");
const tools = {git: "/tools/git", cmake: "/tools/cmake", cc: "/tools/cc", cxx: "/tools/cxx", ninja: "/tools/ninja"};
const sdkPlan = (jobs = 2, platform = /** @type {NodeJS.Platform} */ ("linux")) => hermesSdkPlan("/source/hermes", "/sdk/new-hermes", tools, jobs, platform);
/** @returns {any} */
function resumeFixture(/** @type {import("node:test").TestContext} */ t, target = process.platform) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-resume-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const output = path.join(root, "sdk"), source = path.join(output, "source"), build = path.join(output, "build");
  for (const directory of [source, path.join(source, ".git/objects/info"), build, path.join(output, "logs")]) {fs.mkdirSync(directory, {recursive: true});}
  fs.mkdirSync(path.join(source, "lib/VM"), {recursive: true});
  fs.writeFileSync(path.join(source, "lib/VM/CMakeLists.txt"), "old policy\n");
  fs.writeFileSync(path.join(source, "lib/VM/Math.cpp"), "unchanged source\n");
  const plan = hermesSdkPlan("/input/hermes.git", output, tools, 2, target);
  const graph = {"scripts/build-native-hermes.js": "c".repeat(64), "scripts/build-source-inputs.js": hermesResumeBaseline.sourceInputsCurrent,
    "scripts/build-environment.js": "e".repeat(64), "scripts/native-audit.js": "a".repeat(64),
    "runtime/hermes-host-rejections.patch": "b".repeat(64), ...(target === "win32" ? {"runtime/hermes-clang-cl.patch": "f".repeat(64)} : {})};
  const recipe = {files: {}, graph}, hostPatch = path.join(path.dirname(__dirname), "runtime/hermes-host-rejections.patch"),
    windowsPatch = path.join(path.dirname(__dirname), "runtime/hermes-clang-cl.patch");
  const fingerprintFiles = [...Object.values(tools), hostPatch, ...(target === "win32" ? [windowsPatch] : [])];
  const fingerprints = Object.fromEntries(fingerprintFiles.map(file => [file, "1".repeat(64)]));
  const oldRecipe = {files: {}, graph: {...graph, "scripts/build-native-hermes.js": hermesResumeBaseline.producer,
    ...(target === "win32" ? {"scripts/build-source-inputs.js": hermesResumeBaseline.sourceInputsOld,
      "runtime/hermes-clang-cl.patch": hermesResumeBaseline.windowsPatch} : {})}};
  const oldFingerprints = {...fingerprints, ...(target === "win32" ? {[windowsPatch]: hermesResumeBaseline.windowsPatch} : {})};
  const names = "lib/VM/CMakeLists.txt\0lib/VM/Math.cpp\0", sources = hermesSourceFiles(source, names);
  const artifacts = /** @type {Record<string, {path: string, sha256: string, bytes: number}>} */ ({});
  for (const [name, file] of Object.entries(plan.artifacts)) {
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, name === "compiler" ? "compiler bytes\n" : "!<arch>\narchive\n");
    artifacts[name] = {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size};
  }
  const before = {pins: hermesPins, recipe: oldRecipe, fingerprints: oldFingerprints, sources, configure: plan.configure, compile: plan.compile};
  const result = {version: 1, target, pins: hermesPins, source: plan.source, build: plan.build, artifacts,
    fingerprints: oldFingerprints, sources, recipe: oldRecipe, buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(build, "CMakeCache.txt"), ["CMAKE_HOME_DIRECTORY:INTERNAL=" + source,
    "CMAKE_CACHEFILE_DIR:INTERNAL=" + build, "CMAKE_GENERATOR:INTERNAL=Ninja", ...plan.configure.filter(arg => arg.startsWith("-D")).map(arg => {
      const match = arg.match(/^-D([^:=]+)(?::([^=]+))?=(.*)$/); if (!match) {throw new Error("invalid fixture plan");}
      return match[1] + ":" + (match[2] || "STRING") + "=" + match[3];
    })].join("\n") + "\n");
  fs.writeFileSync(path.join(build, "build.ninja"), "# generated fixture\n");
  fs.writeFileSync(path.join(output, "before.json"), JSON.stringify(before));
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result));
  const git = (/** @type {string[]} */ args) => args.includes("--others") ? "" : names;
  return {root, output, source, build, plan, recipe, fingerprints, names, git, before, result, platform: target};
}

test.describe("Original Static Hermes SDK source production", () => {
  test("the runtime and AOT compiler use the controller's pinned original revision", () => {
    assert.equal(hermesPins.revision, "26fbb82fad39d9d06901adb1d6d630dacc84c877"); assert.ok(Object.isFrozen(hermesPins));
    assert.deepEqual(sdkPlan().patches, ["runtime/hermes-host-rejections.patch"]);
  });
  test("internal builtin JavaScript is native and both JS JIT and N-API are disabled", () => {
    const p = sdkPlan(); for (const flag of ["-DHERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON", "-DHERMESVM_ALLOW_JIT:STRING=0", "-DHERMES_ENABLE_NAPI:BOOL=OFF"]) {assert.ok(p.configure.includes(flag), flag);}
  });
  test("runtime and JSI builds are static release artifacts, not shared application dependencies", () => {
    const p = sdkPlan(); assert.ok(p.configure.includes("-DBUILD_SHARED_LIBS:BOOL=OFF")); assert.ok(p.configure.includes("-DHERMES_BUILD_SHARED_JSI:BOOL=OFF"));
    assert.ok(p.configure.includes("-DCMAKE_BUILD_TYPE:STRING=Release")); assert.match(p.artifacts.runtime, /libhermesvmlean_a\.a$/);
    assert.match(p.artifacts.jsi, /libjsi\.a$/); assert.equal(p.artifacts.compiler, path.join("/sdk/new-hermes", "build/bin/shermes"));
  });
  test("Windows uses the same source recipe plus clang-cl fixes and a static CRT", () => {
    const p = sdkPlan(2, "win32"); assert.deepEqual(p.patches, ["runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"]);
    assert.ok(p.configure.includes("-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded")); assert.match(p.artifacts.compiler, /shermes\.exe$/);
    assert.match(p.artifacts.runtime, /hermesvmlean_a\.lib$/); assert.match(p.artifacts.context, /boost_context\.lib$/);
  });
  test("Windows DateUtil uses standard CRT math only for the Intel host compiler", /** @param {import("node:test").TestContext} t */ t => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/hermes-clang-cl.patch"), "utf8");
    const marker = "diff --git a/lib/VM/CMakeLists.txt b/lib/VM/CMakeLists.txt", start = patch.indexOf(marker);
    assert.ok(start >= 0, "The Windows source patch must own the DateUtil policy");
    const policy = patch.slice(start).split("\n").filter(line => /^\+(?!\+)/.test(line)).map(line => line.slice(1)).join("\n");
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-date-policy-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(path.join(root, "CMakeLists.txt"), `cmake_minimum_required(VERSION 3.20)
project(HermesHostPolicy NONE)
set(WIN32 \${MOM_TEST_WINDOWS})
${policy}
get_source_file_property(date_options JSLib/DateUtil.cpp COMPILE_OPTIONS)
get_source_file_property(other_options JSLib/Math.cpp COMPILE_OPTIONS)
if(MOM_TEST_EXPECT_FLAG)
  if(NOT date_options STREQUAL "-fno-builtin")
    message(FATAL_ERROR "Missing or duplicated DateUtil option")
  endif()
elseif(NOT date_options STREQUAL "NOTFOUND")
  message(FATAL_ERROR "DateUtil policy escaped its Windows/Intel scope")
endif()
if(NOT other_options STREQUAL "NOTFOUND")
  message(FATAL_ERROR "Other Hermes sources must keep their existing optimization")
endif()
`);
    /** @type {Array<[string, string, string, string]>} */
    const variants = [["windows-intel", "ON", "IntelLLVM", "ON"],
      ["windows-clang", "ON", "Clang", "OFF"], ["linux-intel", "OFF", "IntelLLVM", "OFF"]];
    for (const [name, windows, compiler, expected] of variants) {
      // This checks source properties, not a VS installation or compiler build.
      const result = spawnSync("cmake", ["-G", "Ninja", "-S", root, "-B", path.join(root, name), "-DMOM_TEST_WINDOWS=" + windows,
        "-DCMAKE_CXX_COMPILER_ID=" + compiler, "-DMOM_TEST_EXPECT_FLAG=" + expected], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test("resume admission rejects incomplete, changed or path-drifted caches before mutation", /** @param {import("node:test").TestContext} t */ t => {
    const admit = (/** @type {ReturnType<typeof resumeFixture>} */ f) => validateHermesResumeCache(f.output, f.plan, f.recipe, f.fingerprints, f.git, f.platform);
    assert.doesNotThrow(() => admit(resumeFixture(t, "win32")));
    const reject = (/** @type {(f: ReturnType<typeof resumeFixture>) => void} */ edit, /** @type {RegExp} */ pattern) => {
      const f = resumeFixture(t, "win32"); edit(f);
      assert.throws(() => admit(f), pattern);
    };
    reject(f => {f.result.source = "/untrusted/source"; fs.writeFileSync(path.join(f.output, "result.json"), JSON.stringify(f.result));}, /paths, pins or CMake plan/);
    reject(f => {
      f.result.fingerprints[tools.cc] = "9".repeat(64); f.before.fingerprints = f.result.fingerprints;
      fs.writeFileSync(path.join(f.output, "before.json"), JSON.stringify(f.before)); fs.writeFileSync(path.join(f.output, "result.json"), JSON.stringify(f.result));
    }, /fingerprint changed/);
    reject(f => {
      delete f.result.artifacts["context"];
      fs.writeFileSync(path.join(f.output, "result.json"), JSON.stringify(f.result));
    }, /unknown or missing fields/);
    reject(f => {f.recipe.graph["scripts/build-source-inputs.js"] = "0".repeat(64);}, /recipe baseline is not recognized/);
    reject(f => {
      delete f.result.sources["lib/VM/Math.cpp"]; f.before.sources = f.result.sources;
      fs.writeFileSync(path.join(f.output, "before.json"), JSON.stringify(f.before)); fs.writeFileSync(path.join(f.output, "result.json"), JSON.stringify(f.result));
    }, /source inputs are stale/);
    reject(f => fs.writeFileSync(path.join(f.build, "CMakeCache.txt"), "CMAKE_GENERATOR:INTERNAL=Make\n"), /CMake cache differs/);
    reject(f => fs.writeFileSync(path.join(f.output, "unexpected.json"), "{}"), /unknown paths/);
    reject(f => fs.appendFileSync(f.result.artifacts["runtime"].path, "tampered"), /artifact bytes changed/);
    reject(f => {
      f.git = (/** @type {string[]} */ args) => args.includes("--others")
        ? args.includes("--ignored") ? "" : "untracked-shadow.h\0" : f.names;
    }, /unknown files/);
  });
  test("resume source update changes only DateUtil and preserves other source timestamps", /** @param {import("node:test").TestContext} t */ t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-resume-source-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const destination = path.join(root, "cache-source"), candidate = path.join(root, "candidate-source"), names = "lib/VM/CMakeLists.txt\0lib/VM/Math.cpp\0";
    for (const source of [destination, candidate]) {fs.mkdirSync(path.join(source, "lib/VM"), {recursive: true});}
    fs.writeFileSync(path.join(destination, "lib/VM/CMakeLists.txt"), "old DateUtil policy\n");
    fs.writeFileSync(path.join(candidate, "lib/VM/CMakeLists.txt"), "new DateUtil policy\n");
    fs.writeFileSync(path.join(destination, "lib/VM/Math.cpp"), "same source\n");
    fs.writeFileSync(path.join(candidate, "lib/VM/Math.cpp"), "same source\n");
    const oldTime = new Date(1_600_000_000_000); fs.utimesSync(path.join(destination, "lib/VM/Math.cpp"), oldTime, oldTime);
    const beforeMtime = fs.statSync(path.join(destination, "lib/VM/Math.cpp")).mtimeMs;
    const previous = hermesSourceFiles(destination, names), next = hermesSourceFiles(candidate, names);
    assert.deepEqual(copyHermesSourceChanges(destination, candidate, previous, next, "win32"), ["lib/VM/CMakeLists.txt"]);
    assert.equal(fs.statSync(path.join(destination, "lib/VM/Math.cpp")).mtimeMs, beforeMtime);
    assert.equal(fs.readFileSync(path.join(destination, "lib/VM/CMakeLists.txt"), "utf8"), "new DateUtil policy\n");
    assert.throws(() => copyHermesSourceChanges(destination, candidate, previous, {...next, "lib/VM/Math.cpp": "0".repeat(64)}, "win32"), /delta is not the DateUtil policy only/);
  });
  test("Linux resume requires a known producer and unchanged source inventory", /** @param {import("node:test").TestContext} t */ t => {
    const admit = (/** @type {ReturnType<typeof resumeFixture>} */ f) => validateHermesResumeCache(f.output, f.plan, f.recipe, f.fingerprints, f.git, "linux");
    assert.doesNotThrow(() => admit(resumeFixture(t, "linux")));
    const f = resumeFixture(t, "linux"); f.result.recipe.graph["scripts/build-source-inputs.js"] = hermesResumeBaseline.sourceInputsOld;
    f.before.recipe = f.result.recipe;
    fs.writeFileSync(path.join(f.output, "before.json"), JSON.stringify(f.before)); fs.writeFileSync(path.join(f.output, "result.json"), JSON.stringify(f.result));
    assert.throws(() => admit(f), /recipe changed/);

    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-linux-resume-source-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const destination = path.join(root, "cached"), candidate = path.join(root, "candidate"), names = "lib/VM/Math.cpp\0";
    for (const source of [destination, candidate]) {fs.mkdirSync(path.join(source, "lib/VM"), {recursive: true}); fs.writeFileSync(path.join(source, "lib/VM/Math.cpp"), "same source\n");}
    const previous = hermesSourceFiles(destination, names), unchanged = hermesSourceFiles(candidate, names);
    assert.deepEqual(copyHermesSourceChanges(destination, candidate, previous, unchanged, "linux"), []);
    fs.writeFileSync(path.join(candidate, "lib/VM/Math.cpp"), "changed source\n");
    assert.throws(() => copyHermesSourceChanges(destination, candidate, previous, hermesSourceFiles(candidate, names), "linux"), /Linux Hermes resume requires unchanged sources/);
  });
  test("cache resume is opt-in; fresh plans remain unchanged", () => {
    assert.throws(() => buildNativeHermes("/missing", "/new", tools, 2, {MOM_AOT_BUILD_ONLY: "1"}, {resume: true}), /existing writable cache directory/);
    assert.deepEqual(sdkPlan().patches, ["runtime/hermes-host-rejections.patch"]);
    assert.deepEqual(sdkPlan(2, "win32").patches, ["runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"]);
  });
  test("only the required compiler/runtime targets are built with bounded parallelism", () => {
    const p = sdkPlan(3); assert.deepEqual(p.compile.slice(-5), ["shermes", "hermesvmlean_a", "hermesPublic", "jsi", "boost_context"]);
    assert.deepEqual(p.compile.slice(0, 5), [tools.cmake, "--build", p.build, "--parallel", "3"]);
    assert.ok(p.configure.includes("-DCMAKE_MAKE_PROGRAM:FILEPATH=" + tools.ninja));
    assert.doesNotMatch(JSON.stringify(p.configure), /march=native|Debug|node-gyp/);
  });
  test("missing, extra or relative tools fail before a command can run", () => {
    assert.throws(() => hermesSdkPlan("/source", "/output", {...tools, git: "git"}, 2), /explicit absolute/);
    assert.throws(() => hermesSdkPlan("/source", "/output", /** @type {typeof tools} */ (/** @type {unknown} */ ({...tools, extra: "tool"})), 2), /explicit absolute/);
  });
  test("CMake list/interpolation and control characters cannot enter SDK paths", () => {
    for (const tail of [";other", "$ENV{PATH}", '"', "\n"]) {assert.throws(() => hermesSdkPlan("/source" + tail, "/output", tools, 2), /explicit absolute/);}
  });
  test("literal spaces do not require shell commands or quoting fragments", () => {
    const p = hermesSdkPlan("/source SDK", "/new SDK", tools, 2); assert.ok(p.configure.includes(path.join("/new SDK", "source"))); assert.ok(!p.configure.includes("sh"));
  });
  test("unsupported hosts and invalid job counts fail closed", () => {
    assert.throws(() => sdkPlan(2, "darwin"), /Linux\/Windows/);
    for (const jobs of [0, -1, 17, 2.5, NaN, Infinity]) {assert.throws(() => sdkPlan(jobs), /jobs in 1..16/);}
  });
  test("SDK output cannot be inside, replace or contain selected source", () => {
    for (const output of ["/source/hermes", "/source/hermes/build", "/source"]) {assert.throws(() => hermesSdkPlan("/source/hermes", output, tools, 2), /overlap/);}
  });
  test("ambient CMake/Python/compiler hooks are removed without mutating the caller", () => {
    const original = {CMAKE_TOOLCHAIN_FILE: "/bad", CXXFLAGS: "-march=native", PYTHONPATH: "/bad", LD_PRELOAD: "/bad", GIT_CONFIG_COUNT: "1", INCLUDE: "sdk", LIB: "sdk", PATH: "/tools"};
    const result = nativeSdkEnvironment(original);
    for (const key of ["CMAKE_TOOLCHAIN_FILE", "CXXFLAGS", "PYTHONPATH", "LD_PRELOAD", "GIT_CONFIG_COUNT"]) {assert.equal(result[key], undefined);}
    assert.equal(result["INCLUDE"], "sdk"); assert.equal(result["LIB"], "sdk"); assert.equal(original.CMAKE_TOOLCHAIN_FILE, "/bad");
  });
  test("no SDK or compiler is touched without the explicit build-only boundary", () => {
    assert.throws(() => buildNativeHermes("/missing", "/new", tools, 2, {}), /MOM_AOT_BUILD_ONLY=1/);
  });
  test("source inventory rejects escaping and duplicate names before filesystem access", () => {
    for (const names of ["", "../bad\0", "/bad\0", "a\\bad\0", "a\0a\0"]) {assert.throws(() => hermesSourceFiles("/missing", names), /source inventory/);}
  });
  test("tracked-source edits change the actual byte inventory", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-sdk-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(path.join(root, "source.cpp"), "old"); const before = hermesSourceFiles(root, "source.cpp\0");
    fs.writeFileSync(path.join(root, "source.cpp"), "new"); assert.notDeepEqual(hermesSourceFiles(root, "source.cpp\0"), before);
  });
  test("SDK source symlinks cannot read unrelated files", {skip: process.platform === "win32"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-sdk-link-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.symlinkSync("/etc/passwd", path.join(root, "source.cpp")); assert.throws(() => hermesSourceFiles(root, "source.cpp\0"), /regular files/);
  });
});
