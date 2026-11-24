"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {fileHash} = require("../scripts/build-source-inputs");
const {syclSources, dpcppGpuFlags, dpcppSourceFlags} = require("../scripts/native-gpu-build");
const {
  validateWindowsGpuConfig, resolveOneapiCompiler, windowsGpuGit, windowsGpuBuildJobs, windowsGpuSources, windowsGpuHostFlags, windowsGpuDeviceFlags,
  windowsGpuSourceFlags, windowsGpuCompileCommands, windowsHostSections,
  buildNativeWindowsGpu, windowsGpuBuildEnvironment,
} = require("../scripts/build-native-windows-gpu");

const repo = path.resolve(__dirname, "..");
const compilerExecutableSuffix = process.platform === "win32" ? ".exe" : "";
const gyp = fs.readFileSync(path.join(repo, "binding.gyp"), "utf8");
const combinedWrapper = fs.readFileSync(path.join(repo, "scripts/cxx-combined.sh"), "utf8");

test.describe("Windows GPU compiler environment", () => {
  test("CI transport changes do not invalidate compiler objects or reach compiler children", () => {
    const semantic = {PATH: "/compiler/bin", LD_LIBRARY_PATH: "/compiler/lib", HOME: "/tmp",
      MOM_BUILD_JOBS: "2", MOM_GPU_TUNING: "keep", CXXFLAGS: "-O2"};
    const first = {...semantic, RUNNER_TEMP: "/run-one", GITHUB_OUTPUT: "/run-one/output",
      GITHUB_RUN_ID: "1", GITHUB_SHA: "first", MOM_NATIVE_WINDOWS_GPU_DIR: "/output-one"};
    const second = {...semantic, RUNNER_TEMP: "/run-two", GITHUB_OUTPUT: "/run-two/output",
      GITHUB_RUN_ID: "2", GITHUB_SHA: "second", MOM_NATIVE_WINDOWS_GPU_DIR: "/output-two"};
    const a = windowsGpuBuildEnvironment(first), b = windowsGpuBuildEnvironment(second);
    assert.deepEqual(a, b);
    for (const [name, value] of Object.entries(semantic)) {assert.equal(a[name], value);}
    assert.equal(first.RUNNER_TEMP, "/run-one", "The producer retains its transport environment");
    assert.notDeepEqual(a, windowsGpuBuildEnvironment({...first, CXXFLAGS: "-O3"}));
    assert.notDeepEqual(a, windowsGpuBuildEnvironment({...first, LD_LIBRARY_PATH: "/different/lib"}));
    assert.equal(windowsGpuBuildEnvironment({github_output: "ignored", runner_temp: "ignored"})["github_output"], undefined);
  });
});

/** @typedef {"oneapi" | "dpcpp" | "dpcpp-opencl" | "acpp-cuda" | "acpp-hip"} Profile */
/** @typedef {{version: 1, profile: Profile, tools: {clang: string, clangC: string, llvmNm: string, llvmReadobj: string, llvmObjcopy: string}, includeRoots: string[], overlay: string, compilerRoot?: string, cuda?: {sdk: string, nvrtcInclude: string, headers: string}, acpp?: {source: string, generatedInclude: string, plugin: string, git?: string}}} TestConfig */

/** @param {string} name @returns {string} */
function sdkPath(name) {return path.join("/sdk fixture/oneapi", name);}

/** @param {Profile} [profile] @param {boolean} [cuda] @returns {TestConfig} */
function descriptor(profile = "oneapi", cuda = profile === "dpcpp" || profile === "acpp-cuda") {
  const config = /** @type {TestConfig} */ ({
    version: 1,
    profile,
    tools: {
      clang: sdkPath("bin/clang++"),
      clangC: sdkPath("bin/clang"),
      llvmNm: sdkPath("bin/llvm-nm"),
      llvmReadobj: sdkPath("bin/llvm-readobj"),
      llvmObjcopy: sdkPath("bin/llvm-objcopy"),
    },
    includeRoots: [sdkPath("include/clang"), sdkPath("include/sycl")],
    overlay: sdkPath("windows/vfs-overlay.json"),
  });
  if (cuda) {
    config.cuda = {
      sdk: sdkPath("cuda"),
      nvrtcInclude: sdkPath("cuda/include"),
      headers: sdkPath("cuda/nvrtc-headers.inc"),
    };
  }
  if (profile.startsWith("acpp-")) {config.acpp = {source: sdkPath("source"), generatedInclude: sdkPath("generated/include"), plugin: sdkPath("bin/acpp-clang")};}
  return config;
}

/** @param {import("node:test").TestContext} t */
function intelFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-2026.2-")), extension = process.platform === "win32" ? ".exe" : "";
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const config = descriptor("oneapi"); config.compilerRoot = root;
  const helperDirectory = path.join(root, "bin", "compiler"), bin = path.join(root, "bin"), resourceDir = path.join(root, "lib", "clang", "23");
  fs.mkdirSync(helperDirectory, {recursive: true}); fs.mkdirSync(path.join(resourceDir, "include"), {recursive: true});
  config.includeRoots[0] = path.join(resourceDir, "include");
  /** @param {string} file */
  const executable = file => {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture executable\n"); fs.chmodSync(file, 0o755); return file;};
  executable(path.join(bin, "icpx" + extension)); executable(path.join(bin, "icx" + extension));
  const helperNames = ["clang", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
    "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"];
  const helpers = Object.fromEntries(helperNames.map(name => [name, executable(path.join(helperDirectory, name + extension))]));
  const inspectorDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-inspectors-"));
  t.after(() => fs.rmSync(inspectorDirectory, {recursive: true, force: true}));
  const inspectors = Object.fromEntries([["llvmNm", "llvm-nm"], ["llvmReadobj", "llvm-readobj"], ["llvmObjcopy", "llvm-objcopy"]]
    .map(([key, name]) => [key, executable(path.join(inspectorDirectory, name + extension))]));
  const icpxConfig = path.join(bin, "icpx.cfg"), icxConfig = path.join(bin, "icx.cfg");
  fs.writeFileSync(icpxConfig, "compiler config\n"); fs.writeFileSync(icxConfig, "compiler config\n");
  config.tools = {clang: path.join(bin, "icpx" + extension), clangC: path.join(bin, "icx" + extension), ...inspectors};
  /** @param {string} file */
  const compilerVersion = file => [
    "Intel(R) oneAPI DPC++/C++ Compiler 2026.2.0 (2026.2.0.20261001)",
    "Target: x86_64-unknown-linux-gnu", "Thread model: posix", "Configuration file: " + file,
  ].join("\n") + "\n";
  const calls = /** @type {Array<{command: string, args: string[], options: import("node:child_process").SpawnSyncOptions}>} */ ([]);
  /** @type {NonNullable<Parameters<typeof resolveOneapiCompiler>[2]>} */
  const spawn = (command, args, options) => {
    calls.push({command, args, options});
    if (args[0] === "--version") {
      return {status: 0, signal: null, stdout: compilerVersion(command === config.tools.clang ? path.join(bin, "compiler", "..", "icpx.cfg") : path.join(bin, "compiler", "..", "icx.cfg"))};
    }
    if (args[0] === "--print-resource-dir") {return {status: 0, signal: null, stdout: resourceDir + "\n"};}
    if (args[0]?.startsWith("-print-prog-name=")) {
      const name = (args[0].slice("-print-prog-name=".length) || "").replace(/\.exe$/i, "");
      return {status: 0, signal: null, stdout: (helpers[name] || name) + "\n"};
    }
    return {status: 1, signal: null, stdout: ""};
  };
  return {root, config, helpers, calls, spawn, resourceDir, icpxConfig, icxConfig};
}

test.describe("Windows native GPU producer admission", () => {
  test("honors bounded build parallelism on Linux and case-insensitive Windows environments", () => {
    assert.equal(windowsGpuBuildJobs({}), 1);
    assert.equal(windowsGpuBuildJobs({MOM_BUILD_JOBS: ""}), 1);
    for (const jobs of [1, 2, 8, 256]) {assert.equal(windowsGpuBuildJobs({MOM_BUILD_JOBS: String(jobs)}), jobs);}
    assert.equal(windowsGpuBuildJobs({mom_build_jobs: "2"}, "win32"), 2);
    assert.equal(windowsGpuBuildJobs({MOM_BUILD_JOBS: "2", mom_build_jobs: "2"}, "win32"), 2);
    assert.throws(() => windowsGpuBuildJobs({MOM_BUILD_JOBS: "1", mom_build_jobs: "2"}, "win32"), /Conflicting case variants/);
    for (const value of ["0", "-1", "257", "2.5", "1e2", "2;echo bad", "NaN", " 2"]) {
      assert.throws(() => windowsGpuBuildJobs({MOM_BUILD_JOBS: value}), /MOM_BUILD_JOBS/);
    }
  });

  test("accepts the three maintained profiles only with their exact descriptor shape", () => {
    for (const profile of /** @type {Profile[]} */ (["oneapi", "dpcpp-opencl"])) {
      const selected = validateWindowsGpuConfig(descriptor(profile));
      assert.equal(selected.profile, profile);
      assert.deepEqual(Object.keys(selected.tools).sort(), ["clang", "clangC", "llvmNm", "llvmObjcopy", "llvmReadobj"]);
      assert.equal(selected.cuda, undefined);
    }
    const cuda = validateWindowsGpuConfig(descriptor("dpcpp"));
    assert.equal(cuda.profile, "dpcpp");
    assert.deepEqual(Object.keys(cuda.cuda || {}).sort(), ["headers", "nvrtcInclude", "sdk"]);
  });

  test("rejects unknown profiles, extra descriptor keys, and malformed tool paths", () => {
    const unknown = {...descriptor(), profile: "oneapi;dpcpp"};
    assert.throws(() => validateWindowsGpuConfig(unknown), /Invalid Windows GPU descriptor\/profile/);
    const extra = {...descriptor(), unexpected: true};
    assert.throws(() => validateWindowsGpuConfig(extra), /Invalid Windows GPU descriptor\/profile/);
    const control = descriptor(); control.tools.clang = "/sdk/clang\u0001";
    assert.throws(() => validateWindowsGpuConfig(control), /five absolute tool paths/);
    const duplicate = descriptor(); duplicate.includeRoots.push(duplicate.includeRoots[0] || "");
    assert.throws(() => validateWindowsGpuConfig(duplicate), /include roots\/overlay/);
  });

  test("admits compilerRoot only as an optional explicit oneAPI input", () => {
    const selected = descriptor(); selected.compilerRoot = sdkPath("compiler/2026.2");
    assert.equal(validateWindowsGpuConfig(selected).compilerRoot, selected.compilerRoot);
    const relative = {...descriptor(), compilerRoot: "compiler/2026.2"};
    assert.throws(() => validateWindowsGpuConfig(relative), /compilerRoot/);
    for (const profile of /** @type {Profile[]} */ (["dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"])) {
      assert.throws(() => validateWindowsGpuConfig({...descriptor(profile), compilerRoot: sdkPath("compiler/2026.2")}), /only as an absolute oneAPI/);
    }
    assert.equal(validateWindowsGpuConfig(descriptor("oneapi")).compilerRoot, undefined);
  });

  test("resolves and hashes canonical Intel 2026.2 tools and the matching resource include", t => {
    const f = intelFixture(t), selected = resolveOneapiCompiler(f.config, {}, f.spawn);
    assert.equal(selected.root, f.root);
    assert.equal(selected.tools.clang, f.config.tools.clang);
    assert.equal(selected.tools.clangC, f.config.tools.clangC);
    assert.deepEqual([selected.tools.llvmNm, selected.tools.llvmReadobj, selected.tools.llvmObjcopy],
      [f.config.tools.llvmNm, f.config.tools.llvmReadobj, f.config.tools.llvmObjcopy]);
    assert.equal(selected.resourceDir, f.resourceDir);
    assert.equal(selected.resourceInclude, path.join(f.resourceDir, "include"));
    assert.deepEqual(selected.sdkTools, ["clang", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
      "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"].map(name => f.helpers[name] || ""));
    for (const file of [...Object.values(selected.tools), ...selected.sdkTools, f.icpxConfig, f.icxConfig]) {
      assert.equal(typeof selected.inputs[file], "string", file);
    }
    for (const file of [selected.tools.clang, selected.tools.clangC, ...selected.sdkTools, f.icpxConfig, f.icxConfig]) {
      assert.equal(path.relative(f.root, file).startsWith(".."), false, file);
    }
    for (const file of [selected.tools.llvmNm, selected.tools.llvmReadobj, selected.tools.llvmObjcopy]) {
      assert.equal(path.relative(f.root, file).startsWith(".."), true, file);
    }
    assert.ok(f.calls.some(call => call.command === f.config.tools.clang && call.args[0] === "--print-resource-dir"));
    assert.ok(f.calls.some(call => call.command === f.config.tools.clang && call.args[0] === "-print-prog-name=clang-offload-bundler" + compilerExecutableSuffix));
    assert.ok(!f.calls.some(call => ["llvm-nm", "llvm-readobj", "llvm-objcopy"].some(name =>
      call.args[0] === "-print-prog-name=" + name + compilerExecutableSuffix)));
    assert.ok(f.calls.some(call => call.command === f.config.tools.clangC && call.args[0] === "--version"));
    assert.ok(f.calls.some(call => call.command === f.config.tools.clangC && call.args[0] === "--print-resource-dir"));
    assert.ok(f.calls.some(call => call.command === f.config.tools.clangC && call.args[0] === "-print-prog-name=clang-offload-bundler" + compilerExecutableSuffix));
    const identityHelper = path.join(repo, "scripts", "build-native-windows-hiprtc-sdk.js");
    assert.equal(typeof selected.inputs[identityHelper], "string");
  });

  test("requires the queried Intel resource include in the descriptor without rewriting flags", t => {
    const f = intelFixture(t), selected = resolveOneapiCompiler(f.config, {}, f.spawn);
    const host = windowsGpuHostFlags(f.config, "/output"), hostC = windowsGpuHostFlags(f.config, "/output", true);
    const device = windowsGpuDeviceFlags(f.config, host, gyp, combinedWrapper);
    for (const flags of [host, hostC, device]) {
      const includes = flags.flatMap((flag, index) => flag === "-isystem" ? [flags[index + 1] || ""] : []);
      assert.deepEqual(includes.slice(0, f.config.includeRoots.length), f.config.includeRoots);
    }
    assert.equal(f.config.includeRoots[0], selected.resourceInclude);
    const legacy = descriptor(), oldFlags = windowsGpuHostFlags(legacy, "/output");
    assert.ok(oldFlags.includes(legacy.includeRoots[0] || ""));
  });

  test("fails closed on non-Intel versions, unresolved helpers, and helpers outside compilerRoot", t => {
    const wrongVersion = intelFixture(t), normalSpawn = wrongVersion.spawn;
    wrongVersion.spawn = (command, args, options) => args[0] === "--version" && command === wrongVersion.config.tools.clangC
      ? {status: 0, signal: null, stdout: "Clang 2026.2.0\n"} : normalSpawn(command, args, options);
    assert.throws(() => resolveOneapiCompiler(wrongVersion.config, {}, wrongVersion.spawn), /Intel 2026\.2/);

    const unresolved = intelFixture(t), unresolvedSpawn = unresolved.spawn;
    unresolved.spawn = (command, args, options) => args[0] === "-print-prog-name=llc" + compilerExecutableSuffix
      ? {status: 0, signal: null, stdout: "llc\n"} : unresolvedSpawn(command, args, options);
    assert.throws(() => resolveOneapiCompiler(unresolved.config, {}, unresolved.spawn), /absolute path/);

    const outside = intelFixture(t), outsideSpawn = outside.spawn, external = path.join(path.dirname(outside.root), "mom-outside-llc");
    fs.writeFileSync(external, "external helper\n"); fs.chmodSync(external, 0o755); t.after(() => fs.rmSync(external, {force: true}));
    outside.spawn = (command, args, options) => args[0] === "-print-prog-name=llc" + compilerExecutableSuffix
      ? {status: 0, signal: null, stdout: external + "\n"} : outsideSpawn(command, args, options);
    assert.throws(() => resolveOneapiCompiler(outside.config, {}, outside.spawn), /under compilerRoot/);

    const mismatched = intelFixture(t), matchingSpawn = mismatched.spawn;
    mismatched.spawn = (command, args, options) => command === mismatched.config.tools.clangC && args[0] === "-print-prog-name=llc" + compilerExecutableSuffix
      ? {status: 0, signal: null, stdout: (mismatched.helpers["clang"] || "") + "\n"} : matchingSpawn(command, args, options);
    assert.throws(() => resolveOneapiCompiler(mismatched.config, {}, mismatched.spawn), /resolve different helper paths/);

    const staleResource = intelFixture(t); staleResource.config.includeRoots[0] = sdkPath("old-dpcpp/resource/include");
    assert.throws(() => resolveOneapiCompiler(staleResource.config, {}, staleResource.spawn), /includeRoots\[0\]/);
  });

  test("requires CUDA inputs only for the DPC++ CUDA profile", () => {
    for (const profile of /** @type {Profile[]} */ (["oneapi", "dpcpp-opencl"])) {
      const invalid = {...descriptor(profile), cuda: descriptor("dpcpp").cuda};
      assert.throws(() => validateWindowsGpuConfig(invalid), /CUDA inputs are not valid/);
    }
    assert.throws(() => validateWindowsGpuConfig(descriptor("dpcpp", false)), /needs explicit CUDA/);
    const incomplete = descriptor("dpcpp"), incompleteCuda = incomplete.cuda;
    assert.ok(incompleteCuda);
    Object.defineProperty(incompleteCuda, "headers", {value: undefined, enumerable: true});
    assert.throws(() => validateWindowsGpuConfig(incomplete), /needs explicit CUDA/);
    const extra = descriptor("dpcpp"), extraCuda = extra.cuda;
    assert.ok(extraCuda);
    Object.defineProperty(extraCuda, "extra", {value: sdkPath("cuda/extra"), enumerable: true});
    assert.throws(() => validateWindowsGpuConfig(extra), /needs explicit CUDA/);
  });

  test("rejects relative, empty, and overlong SDK path lists", () => {
    const relative = descriptor(); relative.overlay = "windows/vfs-overlay.json";
    assert.throws(() => validateWindowsGpuConfig(relative), /include roots\/overlay/);
    const empty = descriptor(); empty.includeRoots = [];
    assert.throws(() => validateWindowsGpuConfig(empty), /include roots\/overlay/);
    const tooMany = descriptor(); tooMany.includeRoots = Array.from({length: 33}, (_, index) => sdkPath("include/" + index));
    assert.throws(() => validateWindowsGpuConfig(tooMany), /include roots\/overlay/);
  });

  test("keeps oneAPI, CUDA DPC++, and portable OpenCL flags on separate profiles", () => {
    const oneapi = dpcppGpuFlags(gyp, "oneapi"), dpcpp = dpcppGpuFlags(gyp, "dpcpp"), opencl = dpcppGpuFlags(gyp, "dpcpp-opencl");
    assert.ok(oneapi.includes("-DMOM_SYCL_INTEL_LAYOUT"));
    assert.ok(!oneapi.includes("-DMOM_SYCL_HAS_CUDA"));
    assert.ok(dpcpp.includes("-DMOM_SYCL_HAS_CUDA"));
    assert.ok(dpcpp.some(flag => /-fsycl-targets=.*nvidia_gpu_sm_/.test(flag)));
    assert.ok(opencl.includes("-DMOM_SYCL_PORTABLE_OPENCL"));
    assert.ok(!opencl.some(flag => /CUDA|ESIMD|SYCL_NATIVE/.test(flag)));
  });

  test("does not apply the mixed DPC++ ZHash override to oneAPI or Nexa", () => {
    assert.deepEqual(dpcppSourceFlags(combinedWrapper, "sycl/zhash/zhash.cpp", "dpcpp"), ["-DMOM_ZHASH_INTEL_LATE_BUCKETS"]);
    assert.deepEqual(dpcppSourceFlags(combinedWrapper, "sycl/zhash/zhash.cpp", "oneapi"), []);
    assert.deepEqual(dpcppSourceFlags(combinedWrapper, "sycl/nexapow/nexapow.cpp", "oneapi"), []);
  });
});

test.describe("Windows AdaptiveCpp native source policy", () => {
  test("SSCP disables host autolinking only on its retargeted Windows modules", () => {
    for (const profile of /** @type {Profile[]} */ (["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"])) {
      const c = descriptor(profile), host = windowsGpuHostFlags(c, "/output");
      assert.equal(host.includes("-fno-autolink"), false);
      assert.equal(windowsGpuDeviceFlags(c, host, gyp, combinedWrapper).includes("-fno-autolink"), profile.startsWith("acpp-"));
    }
  });
  test("actual Windows host directives fail ELF lowering before the compiler flag, not afterward", {
    skip: process.platform !== "linux" || !fs.existsSync("/usr/lib/llvm-21/bin/llc"), timeout: 30000,
  }, () => {
    const clang = "/usr/lib/llvm-21/bin/clang", llc = "/usr/lib/llvm-21/bin/llc";
    const c = descriptor("acpp-hip"), actual = windowsGpuDeviceFlags(c, windowsGpuHostFlags(c, "/output"), gyp, combinedWrapper).filter(flag => flag === "-fno-autolink");
    assert.deepEqual(actual, ["-fno-autolink"]);
    for (const fixed of [false, true]) {
      const ir = spawnSync(clang, ["--target=x86_64-pc-windows-msvc", "-Xclang", "--dependent-lib=libcmt", ...(fixed ? actual : []), "-x", "c++", "-fms-extensions", "-emit-llvm", "-S", "-o", "-", "-"], {input: '#pragma comment(lib, "host-only.lib")\nint sharedValue;\n', encoding: "utf8", timeout: 5000, maxBuffer: 1 << 20});
      assert.equal(ir.status, 0, ir.error?.message || ir.stderr);
      assert.equal(ir.stdout.includes("!llvm.linker.options"), !fixed);
      assert.match(ir.stdout, /!"wchar_size", i32 2/); // ABI checks remain intact.
      const object = spawnSync("/bin/bash", ["-c", 'ulimit -c 0; exec "$@"', "mom-elf-negative-control", llc, "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx900", "-filetype=obj", "-o", "/dev/null", "-"], {input: ir.stdout, encoding: "utf8", timeout: 5000, maxBuffer: 1 << 20});
      if (fixed) {assert.equal(object.status, 0, object.error?.message || object.stderr);} else {assert.notEqual(object.status, 0); assert.match(object.stderr, /invalid llvm\.linker\.options/);}
    }
  });
  test("admits both isolated profiles with explicit pinned source and plugin paths", () => {
    for (const profile of /** @type {Profile[]} */ (["acpp-cuda", "acpp-hip"])) {
      const config = descriptor(profile); assert.equal(validateWindowsGpuConfig(config), config);
      assert.ok(config.acpp);
    }
  });

  test("Windows checkout admission uses an explicit Git outside the compiler PATH", () => {
    for (const profile of /** @type {Profile[]} */ (["acpp-cuda", "acpp-hip"])) {
      const config = descriptor(profile); assert.ok(config.acpp);
      assert.equal(windowsGpuGit(config, "linux"), "/usr/bin/git");
      assert.throws(() => windowsGpuGit(config, "win32"), /explicit absolute Git/);
      config.acpp.git = sdkPath("git/bin/git.exe");
      assert.equal(windowsGpuGit(validateWindowsGpuConfig(config), "win32"), config.acpp.git);
      config.acpp.git = "relative/git";
      assert.throws(() => validateWindowsGpuConfig(config), /AdaptiveCpp needs explicit/);
    }
    assert.equal(windowsGpuGit(descriptor("oneapi"), "win32"), "");
  });

  test("rejects missing, relative, extra, or cross-profile AdaptiveCpp descriptor fields", () => {
    const missing = descriptor("acpp-hip"); delete missing.acpp;
    assert.throws(() => validateWindowsGpuConfig(missing), /AdaptiveCpp needs explicit/);
    const relative = descriptor("acpp-hip"); assert.ok(relative.acpp); relative.acpp.plugin = "relative/plugin";
    assert.throws(() => validateWindowsGpuConfig(relative), /AdaptiveCpp needs explicit/);
    const extra = descriptor("acpp-hip"); Object.assign(extra.acpp || {}, {extra: sdkPath("extra")});
    assert.throws(() => validateWindowsGpuConfig(extra), /AdaptiveCpp needs explicit/);
    assert.throws(() => validateWindowsGpuConfig({...descriptor(), acpp: descriptor("acpp-hip").acpp}), /not valid for this Windows GPU profile/);
    assert.throws(() => validateWindowsGpuConfig({...descriptor("acpp-hip"), cuda: descriptor("acpp-cuda").cuda}), /CUDA inputs are not valid/);
    assert.throws(() => validateWindowsGpuConfig(descriptor("acpp-cuda", false)), /needs explicit CUDA/);
  });

  test("uses generic embedded IR with static Windows ABI and no DPC++ device link flags", () => {
    for (const profile of /** @type {Profile[]} */ (["acpp-cuda", "acpp-hip"])) {
      const config = descriptor(profile), host = windowsGpuHostFlags(config, "/output"), flags = windowsGpuDeviceFlags(config, host, gyp, combinedWrapper);
      for (const required of ["-D_MT", "-U_DLL", "-DMOM_NATIVE_RUNTIME", "-DACPP_STATIC_RUNTIME", "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", "-D_USE_MATH_DEFINES", "-acpp-sscp"]) {assert.ok(flags.includes(required), required);}
      assert.ok(flags.includes("-fplugin=" + config.acpp?.plugin));
      assert.ok(flags.includes("-fpass-plugin=" + config.acpp?.plugin));
      assert.doesNotMatch(flags.join("\0"), /-fsycl|MOM_EMBEDDED_SYCL_JIT|dpcpp-static|MOM_SYCL_BUILD|-fopenmp/);
    }
  });

  test("preserves original per-source CUDA exceptions without leaking them into HIP or KawPow", () => {
    const cuda = descriptor("acpp-cuda"), hip = descriptor("acpp-hip");
    assert.deepEqual(windowsGpuSourceFlags(cuda, "sycl/cn-gpu/cn-gpu.cpp", combinedWrapper), ["-DMOM_SYCL_HAS_CUDA", "-mllvm", "-acpp-sscp-hcf-object-id-namespace=acpp-cuda"]);
    assert.deepEqual(windowsGpuSourceFlags(cuda, "sycl/pearlhash/pearlhash.cpp", combinedWrapper), ["-DMOM_SYCL_HAS_CUDA", "-DMOM_EMBEDDED_NVRTC_HEADERS", "-mllvm", "-acpp-sscp-hcf-object-id-namespace=acpp-cuda"]);
    for (const file of ["sycl/pearlhash/pearlhash.cpp", "sycl/cn-gpu/cn-gpu.cpp", "sycl/kawpow/kawpow.cpp", "sycl/octopus/octopus.cpp"]) {
      assert.deepEqual(windowsGpuSourceFlags(hip, file, combinedWrapper), ["-mllvm", "-acpp-sscp-hcf-object-id-namespace=acpp-hip"]);
    }
    assert.deepEqual(windowsGpuSourceFlags(cuda, "sycl/kawpow/kawpow.cpp", combinedWrapper), ["-mllvm", "-acpp-sscp-hcf-object-id-namespace=acpp-cuda"]);
    assert.deepEqual(windowsGpuSourceFlags(cuda, "runtime/compiler-gate.cpp", combinedWrapper), []);
    assert.deepEqual(windowsGpuSourceFlags(cuda, "sycl/hoohash/host-math.c", combinedWrapper), []);
    const cudaFlags = windowsGpuDeviceFlags(cuda, [], gyp, combinedWrapper), hipFlags = windowsGpuDeviceFlags(hip, [], gyp, combinedWrapper);
    assert.ok(cudaFlags.includes("-DMOM_SYCL_ADAPTIVECPP_CUDA") && cudaFlags.includes("-DMOM_STATIC_NVRTC"));
    assert.ok(!cudaFlags.includes("-DMOM_SYCL_HAS_CUDA") && !cudaFlags.includes("-DMOM_SYCL_HAS_HIP"));
    assert.ok(hipFlags.includes("-DMOM_SYCL_HAS_HIP") && hipFlags.includes("-DMOM_STATIC_HIPRTC"));
    assert.ok(!hipFlags.includes("-DMOM_STATIC_NVRTC"));
  });

  test("passes profile-specific deterministic HCF namespaces only to ACPP SYCL compile commands", () => {
    for (const profile of /** @type {Profile[]} */ (["acpp-cuda", "acpp-hip"])) {
      const config = descriptor(profile), sourceFiles = ["sycl/octopus/octopus.cpp", "runtime/compiler-gate.cpp"],
        hostFlags = windowsGpuHostFlags(config, "/output"), gpuFlags = windowsGpuDeviceFlags(config, hostFlags, gyp, combinedWrapper),
        commands = windowsGpuCompileCommands({sourceFiles, deviceCount: 1, repo, output: "/output", config, gpuFlags, hostFlags,
          perSourceFlags: [windowsGpuSourceFlags(config, sourceFiles[0] || "", combinedWrapper)]}),
        identity = "-acpp-sscp-hcf-object-id-namespace=" + profile;
      assert.ok(commands[0]?.args.includes(identity), profile + " SYCL command carries its profile namespace");
      assert.equal(commands[0]?.args.filter(flag => flag === identity).length, 1);
      const option = commands[0]?.args.indexOf(identity) ?? -1;
      assert.equal(commands[0]?.args[option - 1], "-mllvm");
      assert.ok(!commands[1]?.args.some(flag => flag.startsWith("-acpp-sscp-hcf-object-id-namespace=")), "support TU stays outside HCF namespace");
    }
    for (const profile of /** @type {Profile[]} */ (["oneapi", "dpcpp", "dpcpp-opencl"])) {
      const config = descriptor(profile), hostFlags = windowsGpuHostFlags(config, "/output"), gpuFlags = windowsGpuDeviceFlags(config, hostFlags, gyp, combinedWrapper),
        command = windowsGpuCompileCommands({sourceFiles: ["sycl/octopus/octopus.cpp"], deviceCount: 1, repo, output: "/output", config, gpuFlags, hostFlags,
          perSourceFlags: [windowsGpuSourceFlags(config, "sycl/octopus/octopus.cpp", combinedWrapper)]})[0];
      assert.ok(!command?.args.some(flag => flag.startsWith("-acpp-sscp-hcf-object-id-namespace=")), profile + " remains unchanged");
    }
  });
});

test.describe("Windows GPU deterministic COFF timestamps", () => {
  test("Intel SYCL uses stable per-source names for the shared host/device object", () => {
    const config = descriptor("oneapi"); config.compilerRoot = sdkPath("compiler");
    const file = "sycl/kawpow/kawpow.cpp", prefix = "-fsycl-unique-prefix=uide1aae4a6a5850362",
      flags = windowsGpuSourceFlags(config, file, combinedWrapper), index = flags.indexOf(prefix);
    assert.ok(index > 0); assert.equal(flags[index - 1], "-Xclang");
    assert.deepEqual(windowsGpuSourceFlags({...config, compilerRoot: sdkPath("relocated")}, file, combinedWrapper), flags);
    assert.ok(!windowsGpuSourceFlags(config, "sycl/zhash/zhash.cpp", combinedWrapper).includes(prefix));
    const prefixes = windowsGpuSources(gyp).main.map(source => windowsGpuSourceFlags(config, source, combinedWrapper)
      .find(flag => flag.startsWith("-fsycl-unique-prefix=")));
    assert.equal(new Set(prefixes).size, prefixes.length);
    assert.ok(prefixes.every(value => /^-fsycl-unique-prefix=uid[0-9a-f]{16}$/.test(value || "")));
  });

  test("DPC++ and OpenCL use stable profile-specific cc1 prefixes last in each Windows TU", () => {
    const files = windowsGpuSources(gyp).main;
    for (const profile of /** @type {Profile[]} */ (["dpcpp", "dpcpp-opencl"])) {
      const config = descriptor(profile);
      const prefixes = files.map(file => {
        const flags = windowsGpuSourceFlags(config, file, combinedWrapper), prefix = flags.at(-1) || "";
        const expected = "uid" + createHash("sha256").update("mom-sycl-unique-prefix-v1\n" + profile + "\n" + file + "\n").digest("hex").slice(0, 16);
        assert.equal(flags.at(-2), "-Xclang");
        assert.equal(prefix, "-fsycl-unique-prefix=" + expected);
        return prefix;
      });
      assert.equal(new Set(prefixes).size, files.length, profile + " prefixes are unique across device TUs");

      const file = files[0] || "", hostFlags = windowsGpuHostFlags(config, "/output"), gpuFlags = windowsGpuDeviceFlags(config, hostFlags, gyp, combinedWrapper),
        command = windowsGpuCompileCommands({sourceFiles: [file], deviceCount: 1, repo, output: "/output", config, gpuFlags, hostFlags,
          perSourceFlags: [windowsGpuSourceFlags(config, file, combinedWrapper)]})[0]?.args || [],
        prefixIndex = command.lastIndexOf("-Xclang");
      assert.equal(command[prefixIndex + 1], prefixes[0]);
      assert.equal(command[prefixIndex + 2], "-MD", "the explicit prefix is the last compile flag and wins the cc1 ordering");
    }
  });

  for (const profile of /** @type {Profile[]} */ (["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"])) {
    for (const c of [false, true]) {
      test(`${profile} ${c ? "C" : "C++"} host argv disables time-based COFF stamps`, () => {
        const flags = windowsGpuHostFlags(descriptor(profile), "/external/output", c);
        assert.equal(flags.filter(flag => flag === "-mno-incremental-linker-compatible").length, 1);
      });
    }
  }

  test("selected GNU-style driver emits byte-identical zero-stamp COFF in fresh output roots", {
    skip: !process.env["MOM_TEST_WINDOWS_GPU_COMPILER"] && "Set MOM_TEST_WINDOWS_GPU_COMPILER to a pinned compiler path",
    timeout: 30000,
  }, async t => {
    const compiler = process.env["MOM_TEST_WINDOWS_GPU_COMPILER"] || "";
    assert.ok(path.isAbsolute(compiler), "The selected compiler path must be absolute");
    const beforeStat = fs.statSync(compiler);
    assert.ok(beforeStat.isFile(), "The selected compiler must be a regular file");
    const beforeHash = fileHash(compiler), root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-gpu-coff-repro-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const source = path.join(root, "fixture.cpp");
    fs.writeFileSync(source, "int mom_coff_repro_fixture(void) { return 7; }\n", {flag: "wx"});
    const outputDirs = [path.join(root, "a"), path.join(root, "nested", "different", "depth", "b")];
    for (const dir of outputDirs) {fs.mkdirSync(dir, {recursive: true});}
    const publicFlags = windowsGpuHostFlags(descriptor("dpcpp"), root);
    const target = publicFlags.find(flag => flag.startsWith("--target="));
    const optimization = publicFlags.find(flag => /^-O[0-3]$/.test(flag));
    const deterministic = publicFlags.filter(flag => flag === "-mno-incremental-linker-compatible");
    assert.ok(target, "The public host recipe must provide the target triple");
    assert.ok(optimization, "The public host recipe must provide an optimization level");
    assert.ok(deterministic.length <= 1, "The public host recipe must not duplicate its deterministic COFF flag");
    /** @type {string[]} */
    const outputs = [];
    for (const [index, directory] of outputDirs.entries()) {
      if (index) {await new Promise(resolve => setTimeout(resolve, 1250));}
      const file = path.join(directory, "fixture.obj");
      /** @type {import("node:child_process").SpawnSyncReturns<string>} */
      const result = spawnSync(compiler, [target, optimization, ...deterministic, "-x", "c++", "-c", source, "-o", file], {
        cwd: root, encoding: "utf8", timeout: 15000, maxBuffer: 64 << 10,
      });
      assert.equal(result.error, undefined, result.error?.message);
      assert.equal(result.signal, null, result.stderr || "compiler was signaled");
      assert.equal(result.status, 0, (result.stdout || "") + (result.stderr || ""));
      const stat = fs.lstatSync(file);
      assert.ok(stat.isFile() && !stat.isSymbolicLink() && stat.size >= 20 && stat.size <= 64 << 10,
        "The probe must produce a bounded regular COFF object");
      const object = fs.readFileSync(file), sections = object.readUInt16LE(2);
      assert.equal(object.readUInt16LE(0), 0x8664, "Expected standard x64 COFF, not PE or BigObj");
      assert.ok(sections > 0 && sections <= 32 && 20 + sections * 40 <= object.length, "Invalid COFF section table");
      assert.equal(object.readUInt16LE(16), 0, "COFF object must not have a PE optional header");
      assert.equal(object.readUInt32LE(4), 0, "The public flag must zero COFF TimeDateStamp");
      outputs.push(fileHash(file));
    }
    assert.equal(fileHash(compiler), beforeHash, "Compiler bytes changed during the probe");
    assert.equal(outputs[0], outputs[1],
      "The same source and flags must be byte-identical across a time delay and different output depths");
  });
});

test.describe("Windows native GPU source and host policy", () => {
  test("derives the 21-device source inventory from binding.gyp without duplicate support", () => {
    const maintained = windowsGpuSources(gyp), main = syclSources(gyp);
    assert.deepEqual(maintained.main, main);
    assert.equal(maintained.main.length, 21);
    assert.deepEqual(maintained.support.sort(), [
      "sycl/c29/blake2b.cpp", "sycl/etchash/ethash.cpp", "sycl/hoohash/host-math.c",
      "xmrig/base/crypto/keccak.cpp", "xmrig/base/crypto/sha3.cpp",
      "xmrig/crypto/randomx/blake2/blake2b.c",
    ].sort());
    assert.equal(new Set([...maintained.main, ...maintained.support]).size,
      maintained.main.length + maintained.support.length);
    // The development builders retain stable object names, so compare their source paths alone.
    const windowsSources = fs.readFileSync(path.join(repo, "scripts/windows-sycl-sources.ps1"), "utf8");
    const development = [...windowsSources.matchAll(/'([^']+\.cpp)'/g)]
      .map(match => (match[1] || "").replaceAll("\\", "/"));
    assert.deepEqual(development.sort(), [...maintained.main,
      ...maintained.support.filter(file => file.startsWith("sycl/") && file.endsWith(".cpp"))].sort());
  });

  test("keeps C support on the strict host recipe while C++ gets the static header", () => {
    const config = descriptor(), output = "/external/mom windows gpu";
    const cpp = windowsGpuHostFlags(config, output), c = windowsGpuHostFlags(config, output, true);
    for (const flags of [cpp, c]) {
      for (const required of ["-D_MT", "-U_DLL", "-DMOM_NATIVE_RUNTIME", "-nostdinc", "-ffunction-sections", "-fdata-sections"]) {
        assert.ok(flags.includes(required), required);
      }
      assert.ok(flags.includes("-Xclang") && flags.includes("--dependent-lib=libcmt"));
      assert.doesNotMatch(flags.join("\0"), /(?:node|napi|NODE_MODULE|(?:^|\0)-DLL(?:\0|$))/i);
      assert.ok(!flags.includes("-fsycl-host-only"));
    }
    assert.ok(cpp.includes("-std=c++20"));
    assert.ok(cpp.includes("-include"));
    assert.ok(cpp.includes(path.join(repo, "tests/native/dpcpp-static/static-windows.h")));
    assert.ok(!c.includes("-std=c++20"));
    assert.ok(!c.includes("-include"));
  });

  test("routes no-builtin only to the admitted Intel oneAPI mixed-TU host half", t => {
    const f = intelFixture(t), selected = resolveOneapiCompiler(f.config, {}, f.spawn);
    assert.equal(selected.tools.clang, f.config.tools.clang, "the selector is gated by the admitted icpx path");
    const sources = windowsGpuSources(gyp), output = "/external/output";
    const hostFlags = windowsGpuHostFlags(f.config, output);
    assert.ok(!hostFlags.includes("-Xarch_host") && !hostFlags.includes("-fno-builtin"), "the shared device/host prefix stays unchanged");
    const gpuFlags = windowsGpuDeviceFlags(f.config, hostFlags, gyp, combinedWrapper);
    assert.deepEqual(gpuFlags.slice(-2), ["-Xarch_host", "-fno-builtin"]);
    const commands = windowsGpuCompileCommands({sourceFiles: [...sources.main, ...sources.support], deviceCount: sources.main.length,
      repo, output, config: f.config, gpuFlags, hostFlags,
      perSourceFlags: sources.main.map(file => windowsGpuSourceFlags(f.config, file, combinedWrapper))});
    const mixed = commands.find(command => command.source === path.join(repo, "sycl/hoohash/hoohash.cpp"));
    assert.ok(mixed, "the proven HooHash mixed-SYCL source is present");
    const selector = mixed.args.indexOf("-Xarch_host");
    assert.deepEqual(mixed.args.slice(selector, selector + 2), ["-Xarch_host", "-fno-builtin"],
      "the exact argv handed to the object-cache batch carries an adjacent host-only selector pair");
    assert.equal(mixed.command, selected.tools.clang);
    assert.ok(mixed.args.includes("-fsycl"));
    assert.equal(mixed.args.filter(flag => flag === "-fno-builtin").length, 1, "no bare device-visible flag is added");
    assert.ok(!mixed.args.includes("-i_no-use-libirc"), "plain-host runtime policy never enters the mixed SYCL command");

    const cryptoSupport = commands.find(command => command.source === path.join(repo, "sycl/c29/blake2b.cpp"));
    assert.ok(cryptoSupport);
    assert.ok(!cryptoSupport.args.includes("-Xarch_host") && !cryptoSupport.args.includes("-fno-builtin"),
      "algorithm support TUs keep their existing host-only flags");
    assert.ok(cryptoSupport.args.includes("-i_no-use-libirc"), "plain Intel C++ support uses CRT memory calls, not proprietary helpers");
    const cSupport = commands.find(command => command.source.endsWith(path.join("blake2", "blake2b.c")));
    assert.ok(cSupport?.args.includes("-i_no-use-libirc"), "the same policy covers plain Intel C support");
    const hostMath = commands.find(command => command.source === path.join(repo, "sycl/hoohash/host-math.c"));
    assert.ok(hostMath?.args.includes("-fno-builtin"), "the existing strict HooHash host C policy remains intact");
    assert.ok(!hostMath?.args.includes("-Xarch_host"));

    for (const config of [descriptor("oneapi"), descriptor("dpcpp"), descriptor("dpcpp-opencl"), descriptor("acpp-cuda"), descriptor("acpp-hip")]) {
      const flags = windowsGpuDeviceFlags(config, windowsGpuHostFlags(config, output), gyp, combinedWrapper);
      assert.ok(!flags.includes("-Xarch_host") && !flags.includes("-fno-builtin"), config.profile + " must not receive the Intel oneAPI host selector");
      const support = windowsGpuCompileCommands({sourceFiles: sources.support, deviceCount: 0, repo, output, config,
        gpuFlags: flags, hostFlags: windowsGpuHostFlags(config, output), perSourceFlags: []});
      assert.ok(support.every(command => !command.args.includes("-i_no-use-libirc")),
        config.profile + " without an admitted Intel frontend must not receive an Intel-only option");
    }
  });

  test("passes VFS and include roots as literal argv values", () => {
    const config = descriptor(), overlay = "/sdk fixture/$HOME;$(touch nope)/vfs overlay.json";
    const roots = ["/sdk fixture/include one", "/sdk fixture/include;two"];
    config.overlay = overlay; config.includeRoots = roots;
    const flags = windowsGpuHostFlags(config, "/external/output");
    const overlayIndex = flags.indexOf("-ivfsoverlay");
    assert.deepEqual(flags.slice(overlayIndex - 1, overlayIndex + 3), ["-Xclang", "-ivfsoverlay", "-Xclang", overlay]);
    for (const root of roots) {
      const index = flags.findIndex((flag, offset) => flag === "-isystem" && flags[offset + 1] === root);
      assert.ok(index >= 0, root);
    }
    assert.ok(flags.includes(overlay));
    assert.ok(!flags.includes("$HOME"));
    assert.ok(!flags.includes("$(touch nope)"));
  });
});

test.describe("Windows native GPU offload-container admission", () => {
  const valid = [
    "  Name: __CLANG_OFFLOAD_BUNDLE__host-x86_64-pc-windows-msvc (COFF)",
    "  Name: __CLANG_OFFLOAD_BUNDLE__sycl-spir64 (COFF)",
  ].join("\n");

  test("accepts the maintained host and device container pair", () => {
    assert.deepEqual(windowsHostSections(valid), [
      "__CLANG_OFFLOAD_BUNDLE__host-x86_64-pc-windows-msvc",
      "__CLANG_OFFLOAD_BUNDLE__sycl-spir64",
    ]);
  });

  test("accepts Intel's tested msvc19.33.0 host target and triple-qualified SPIR-V image", () => {
    const intel = [
      "  Name: __CLANG_OFFLOAD_BUNDLE__host-x86_64-pc-windows-msvc19.33.0 (COFF)",
      "  Name: __CLANG_OFFLOAD_BUNDLE__sycl-spir64-unknown-unknown (COFF)",
    ].join("\n");
    assert.deepEqual(windowsHostSections(intel), [
      "__CLANG_OFFLOAD_BUNDLE__host-x86_64-pc-windows-msvc19.33.0",
      "__CLANG_OFFLOAD_BUNDLE__sycl-spir64-unknown-unknown",
    ]);
  });

  test("rejects missing, duplicate, and unknown offload containers", () => {
    assert.throws(() => windowsHostSections(valid.replace("__CLANG_OFFLOAD_BUNDLE__host-x86_64-pc-windows-msvc", "__CLANG_OFFLOAD_BUNDLE__sycl-spir64")), /Unrecognized/);
    assert.throws(() => windowsHostSections(valid + "\n  Name: __CLANG_OFFLOAD_BUNDLE__sycl-spir64 (COFF)"), /Unrecognized/);
    assert.throws(() => windowsHostSections(valid.replace("__CLANG_OFFLOAD_BUNDLE__sycl-spir64", "__CLANG_OFFLOAD_BUNDLE__other (COFF)")), /Unrecognized/);
    for (const suffix of ["19.33", "19..33.0", "19.33.0.bad", "19.33.0.1"]) {
      assert.throws(() => windowsHostSections(valid.replace("msvc", "msvc" + suffix)), /Unrecognized/);
    }
  });
});

test.describe("Windows native GPU build boundary", () => {
  test("admits CUDA assembly and image tools before compilation or output creation", {skip: !["linux", "win32"].includes(process.platform)}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-gpu-assembler-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const config = descriptor("dpcpp"), bin = path.join(root, "bin"), suffix = process.platform === "win32" ? ".exe" : "";
    fs.mkdirSync(bin);
    for (const key of Object.keys(config.tools)) {Object.assign(config.tools, {[key]: path.join(bin, key + suffix)});}
    for (const file of [...Object.values(config.tools), ...["clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper", "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"].map(name => path.join(bin, name + suffix))]) {
      fs.writeFileSync(file, "fixture", {flag: "wx"});
    }
    config.overlay = path.join(root, "overlay.json"); fs.writeFileSync(config.overlay, "{}", {flag: "wx"});
    config.cuda = {sdk: path.join(root, "cuda"), nvrtcInclude: path.join(root, "nvrtc"), headers: path.join(root, "nvrtc-headers.inc")};
    for (const file of [path.join(config.cuda.sdk, "include/cuda.h"), path.join(config.cuda.sdk, "nvvm/libdevice/libdevice.10.bc"), path.join(config.cuda.nvrtcInclude, "nvrtc.h"), config.cuda.headers]) {
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture", {flag: "wx"});
    }
    const {headerResources} = require("../scripts/native-jit-headers");
    const requiredHeaders = ["cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"];
    const {source, ...metadata} = headerResources(requiredHeaders.map(name => ({name, data: Buffer.from("fixture")})));
    fs.writeFileSync(config.cuda.headers, source);
    fs.writeFileSync(config.cuda.headers + ".json", JSON.stringify({...metadata, requiredHeaders}));
    const descriptorFile = path.join(root, "sdk.json"), output = path.join(root, "output");
    fs.writeFileSync(descriptorFile, JSON.stringify(config), {flag: "wx"});
    const assembler = path.join(config.cuda.sdk, "bin", "ptxas" + suffix); fs.mkdirSync(path.dirname(assembler)); fs.writeFileSync(assembler, "fixture", {flag: "wx"});
    assert.throws(() => buildNativeWindowsGpu(descriptorFile, output, {MOM_AOT_BUILD_ONLY: "1"}), /fatbinary/);
    assert.equal(fs.existsSync(output), false);
    fs.renameSync(assembler, assembler + ".unused"); fs.writeFileSync(path.join(config.cuda.sdk, "bin", "fatbinary" + suffix), "fixture", {flag: "wx"});
    assert.throws(() => buildNativeWindowsGpu(descriptorFile, output, {MOM_AOT_BUILD_ONLY: "1"}), /ptxas/);
    assert.equal(fs.existsSync(output), false);
  });

  test("freezes source before policy reads and admits every profile flag before output mutation", () => {
    const body = buildNativeWindowsGpu.toString(), snapshot = body.indexOf('nativeSourceInputs(repo, "gpu")'),
      policyRead = body.indexOf('fs.readFileSync(path.join(repo, "binding.gyp")'), mutation = body.indexOf("fs.mkdirSync(output)");
    assert.ok(snapshot >= 0 && policyRead > snapshot && mutation > policyRead);
    for (const token of ["perSourceFlags =", "windowsGpuDeviceFlags(", "windowsGpuSourceFlags(", "windowsDeviceLinkArgs("]) {
      const index = body.indexOf(token); assert.ok(index > snapshot && index < mutation, token);
    }
  });

  test("rejects a non-build-only invocation before touching descriptor, output, or tools", {skip: !["linux", "win32"].includes(process.platform)}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-gpu-admission-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const config = path.join(root, "missing-sdk.json"), output = path.join(root, "new-output");
    assert.throws(() => buildNativeWindowsGpu(config, output, {MOM_AOT_BUILD_ONLY: "0"}), /requires MOM_AOT_BUILD_ONLY/);
    assert.equal(fs.existsSync(output), false);
    assert.equal(fs.existsSync(config), false);
  });
});
