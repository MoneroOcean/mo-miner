"use strict";

// Build-only core proof. Normal Engineering/release builds never invoke this experiment.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {nativeDependencies, auditNativeGlibc, auditNativeSymbols} = require("./native-audit");
const pins = Object.freeze({source: "190878e023dd0979199786b19604b377229a4996",
  emhash: "5e131ba09a5290823fe71099d9c35eb5df5345b6", umf: "5a986843a2469d3e04771c1f27314ba6cc3c62f6",
  levelZero: "b77ced661af97104f5cda03d49596274392f6aa4", computeHeaders: "82aab87fc932edc0558a0302d545a5bcc22edf41",
  hwloc: "32b399232a27ee04f317d743db44e7ba3e154d05",
  openclLoader: "ad770a1b64c6b8d5f2ed4e153f22e4f45939f27f", openclHeaders: "8a97ebc88daa3495d6f57ec10bb515224400186f",
  sdk: "7b2e774121370132f930db508196c5c4abdc6c7763867c9da26f54e5145b881c"});

/** CMake strings must not turn a selected path into a list or an expanded expression.
 * @param {string | undefined} value @returns {value is string} */
function absoluteInput(value) {
  return typeof value === "string" && path.isAbsolute(value) && !/[;"$\\]/.test(value) &&
    !Array.from(value).some((char) => char.charCodeAt(0) < 32);
}

/** An adapter is explicit and build-only; incomplete optional inputs never enable discovery.
 * @param {NodeJS.ProcessEnv} env */
function adapterInputs(env) {
  const adapter = env["MOM_AOT_DPCPP_ADAPTER"] || "none";
  const roots = {levelZero: env["MOM_AOT_DPCPP_LEVEL_ZERO_SOURCE"], computeHeaders: env["MOM_AOT_DPCPP_COMPUTE_HEADERS"],
    hwloc: env["MOM_AOT_DPCPP_HWLOC_SOURCE"]};
  // CUDA also needs the pinned hwloc allocator source; that input must not enable Level Zero.
  if (adapter === "none" && roots.levelZero === undefined && roots.computeHeaders === undefined &&
      (roots.hwloc === undefined || (env["MOM_AOT_DPCPP_CUDA_SDK"] !== undefined && absoluteInput(roots.hwloc)))) {return null;}
  if (adapter !== "level_zero" || !Object.values(roots).every(absoluteInput)) {
    throw new Error("Select level_zero explicitly with all three absolute pinned source roots, or none without adapter roots");
  }
  return /** @type {{levelZero: string, computeHeaders: string, hwloc: string}} */ (roots);
}

/** The optional compiler is build output with a private namespace, never a runtime SDK path.
 * @param {NodeJS.ProcessEnv} env */
function sourceJitInput(env) {
  const object = env["MOM_AOT_DPCPP_JIT_OBJECT"];
  if (object === undefined) {return null;}
  if (!absoluteInput(object) || !object.endsWith(".o")) {throw new Error("Select an absolute isolated source-JIT .o object, or omit it");}
  return object;
}

/** CUDA is an independent opt-in so the same host graph may contain Level Zero as well.
 * @param {NodeJS.ProcessEnv} env */
function cudaSdkInput(env) {
  const sdk = env["MOM_AOT_DPCPP_CUDA_SDK"];
  if (sdk === undefined) {return null;}
  if (!absoluteInput(sdk)) {throw new Error("Select an absolute CUDA SDK root, or omit it");}
  return sdk;
}

/** Static OpenCL needs its original loader AND headers; never use an installed application SO.
 * @param {NodeJS.ProcessEnv} env */
function openclInputs(env) {
  const roots = {openclLoader: env["MOM_AOT_DPCPP_OPENCL_SOURCE"], openclHeaders: env["MOM_AOT_DPCPP_OPENCL_HEADERS"]};
  if (Object.values(roots).every((value) => value === undefined)) {return null;}
  if (!Object.values(roots).every(absoluteInput)) {throw new Error("Select both absolute pinned OpenCL source/header roots, or omit both");}
  return /** @type {{openclLoader: string, openclHeaders: string}} */ (roots);
}

/** Match the source/runtime ABI to the existing Engineering SDK, not whichever clang is on PATH.
 * The receipt identifies the original archive; it is not authentication of arbitrary user SDKs.
 * @param {string} version @param {string} receipt */
function sdkIdentity(version, receipt) {
  if (!version.includes(`clang version 23.0.0git (https://github.com/intel/llvm ${pins.source})`) || receipt.trim() !== pins.sdk) {
    throw new Error("Use the matching pinned DPC++ Engineering SDK and archive receipt");
  }
}

/** @param {string} command @param {string[]} args @param {number} [maxBuffer] */
function capture(command, args, maxBuffer = 1 << 20) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000, maxBuffer});
  if (result.error || result.signal || result.status !== 0) {throw new Error(`Input query failed: ${command}`, {cause: result.error});}
  return result.stdout;
}

/** Audit every proof executable, including the real UMF allocation path, without executing it.
 * A fresh receipt directory preserves earlier evidence across incremental rebuilds.
 * @param {string} output @param {boolean} levelZero @param {boolean} [sourceJit] @param {boolean} [cuda] @param {boolean} [opencl] */
function auditBuild(output, levelZero, sourceJit = false, cuda = false, opencl = false) {
  const audit = fs.mkdtempSync(path.join(output, "audit-"));
  const staticAdapters = levelZero || cuda || opencl;
  const artifacts = ["native-dpcpp-core", ...(staticAdapters ? ["native-static-adapter-selection"] : []), ...(levelZero ? ["native-umf-static", "native-level-zero-driver"] : []),
    ...(opencl ? ["native-opencl-binding-static", "native-opencl-binding-dynamic", "native-opencl-driver"] : [])].map((name) => {
    const executable = path.join(output, name), dependencies = capture("readelf", ["-d", executable]);
    const versions = capture("readelf", ["--version-info", executable]);
    const symbols = capture("nm", ["-gC", executable], 64 * 1024 * 1024);
    fs.writeFileSync(path.join(audit, name + ".dependencies.log"), dependencies);
    fs.writeFileSync(path.join(audit, name + ".symbols.log"), symbols);
    fs.writeFileSync(path.join(audit, name + ".versions.log"), versions);
    auditNativeSymbols(symbols);
    return {executable, bytes: fs.statSync(executable).size,
      sha256: createHash("sha256").update(fs.readFileSync(executable)).digest("hex"), needed: nativeDependencies(dependencies),
      glibc: auditNativeGlibc(versions, "2.39")};
  });
  const repo = path.resolve(__dirname, "..");
  const inputs = ["scripts/probe-static-dpcpp.js", "scripts/native-audit.js", "tests/native/dpcpp-static/CMakeLists.txt",
    "tests/reference/native-dpcpp-static.cpp", "runtime/driver-library.cpp",
    ...(sourceJit ? ["runtime/dpcpp-static-jit.patch"] : []),
    ...(opencl ? ["runtime/dpcpp-static-opencl.patch", "tests/reference/native-opencl-binding.cpp", "tests/native/driver-library.cpp"] : []),
    ...(staticAdapters ? ["runtime/dpcpp-static-adapters.patch", "tests/reference/native-static-adapter-selection.cpp"] : []),
    ...(cuda ? ["runtime/system-driver.h", "scripts/native-cuda-adapter.js",
      "scripts/native-compiler-isolation.js", "tests/native/dpcpp-static/cuda-factory.cpp"] : []),
    ...(levelZero ? ["runtime/umf-static-level-zero.patch", "runtime/level-zero-static-complete.patch",
      "scripts/native-level-zero-adapter.js", "scripts/native-compiler-isolation.js",
      "tests/native/dpcpp-static/level-zero-v2-factory.cpp",
      "tests/reference/native-umf-static.cpp", "tests/reference/native-level-zero-static.cpp"] : [])];
  const sources = Object.fromEntries(inputs.map((name) => [name,
    createHash("sha256").update(fs.readFileSync(path.join(repo, name))).digest("hex")]));
  const receipt = {audit, artifacts, sources};
  fs.writeFileSync(path.join(audit, "result.json"), JSON.stringify(receipt, null, 2) + "\n");
  return receipt;
}

function main() {
  if (process.env["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("This proof requires MOM_AOT_BUILD_ONLY=1; no GPU execution is implicit");}
  const args = process.argv.slice(2);
  if (process.platform !== "linux" || args.length !== 5 || !args.every(absoluteInput)) {
    throw new Error("Usage: probe-static-dpcpp.js <absolute DPC++ source> <SDK> <emhash source> <UMF source> <new output directory>");
  }
  const [source, sdk, emhash, umf, output] = /** @type {[string, string, string, string, string]} */ (args);
  const adapter = adapterInputs(process.env);
  const jitObject = sourceJitInput(process.env);
  const cudaSdk = cudaSdkInput(process.env);
  const opencl = openclInputs(process.env);
  const hwloc = adapter?.hwloc || process.env["MOM_AOT_DPCPP_HWLOC_SOURCE"];
  if (cudaSdk && !absoluteInput(hwloc)) {throw new Error("Static CUDA allocation needs an absolute pinned hwloc source");}
  const repo = path.resolve(__dirname, "..");
  /** @type {Array<[string, string]>} */
  const sources = [[source, pins.source], [emhash, pins.emhash], [umf, pins.umf]];
  if (opencl) {sources.push([opencl.openclLoader, pins.openclLoader], [opencl.openclHeaders, pins.openclHeaders]);}
  if (adapter) {
    for (const [name, directory] of Object.entries(adapter)) {
      sources.push([directory, pins[/** @type {"levelZero" | "computeHeaders" | "hwloc"} */ (name)]]);
    }
  }
  if (cudaSdk && !adapter) {sources.push([/** @type {string} */ (hwloc), pins.hwloc]);}
  for (const [directory, revision] of sources) {
    const difference = capture("git", ["--no-optional-locks", "-C", directory, "diff", "--no-ext-diff", "--no-textconv", "HEAD"]);
    const sourcePatches = [...(jitObject ? ["dpcpp-static-jit.patch"] : []), ...(adapter || cudaSdk || opencl ? ["dpcpp-static-adapters.patch"] : []),
      ...(opencl ? ["dpcpp-static-opencl.patch"] : [])];
    // Feature patches touch disjoint files, but git orders the combined diff by path. Preserve
    // every byte while ordering file blocks; never weaken the exact reviewed-source check.
    const patchedSource = sourcePatches.flatMap((name) => fs.readFileSync(path.join(repo, "runtime", name), "utf8")
      .split(/(?=^diff --git )/m).filter(Boolean)).sort().join("");
    const expected = adapter && directory === umf ? fs.readFileSync(path.join(repo, "runtime/umf-static-level-zero.patch"), "utf8") :
      adapter && directory === adapter.levelZero ? fs.readFileSync(path.join(repo, "runtime/level-zero-static-complete.patch"), "utf8") :
        directory === source ? patchedSource : "";
    if (capture("git", ["--no-optional-locks", "-C", directory, "rev-parse", "HEAD"]).trim() !== revision ||
        difference.trim() !== expected.trim() ||
        // Ignored headers/configuration can affect compilation too. Keep generated build
        // trees outside pinned sources; collapse unknown directories instead of scanning them.
        capture("git", ["--no-optional-locks", "-C", directory, "ls-files", "--others", "--directory", "--"]).length !== 0) {
      throw new Error("Use pinned source inputs with only the reviewed patches for the explicitly selected static features");
    }
  }
  const cxx = path.join(sdk, "bin/clang++"), version = capture(cxx, ["--version"]);
  sdkIdentity(version, fs.readFileSync(path.join(sdk, ".mom-toolchain-sha256"), "utf8"));
  fs.mkdirSync(output); // Preserve prior evidence and never touch Engineering's build/cache tree.
  fs.writeFileSync(path.join(output, "compiler-version.log"), version);
  /** @param {string} command @param {string[]} commandArgs @param {string} name */
  function build(command, commandArgs, name) {
    const fd = fs.openSync(path.join(output, name + ".log"), "wx");
    try {
      const result = spawnSync(command, commandArgs, {stdio: ["ignore", fd, fd], timeout: 600000});
      if (result.error || result.signal || result.status !== 0) {throw new Error(`Static DPC++ build failed: ${name}.log`, {cause: result.error});}
    } finally {fs.closeSync(fd);}
  }
  const definitions = {MOM_DPCPP_SOURCE: source, MOM_DPCPP_SDK: sdk, MOM_DPCPP_EMHASH: emhash,
    MOM_DPCPP_UMF: umf, MOM_SOURCE_REPO: repo, CMAKE_CXX_COMPILER: cxx, CMAKE_C_COMPILER: path.join(sdk, "bin/clang")};
  if (opencl) {Object.assign(definitions, {MOM_DPCPP_OPENCL_SOURCE: opencl.openclLoader, MOM_DPCPP_OPENCL_HEADERS: opencl.openclHeaders});}
  if (cudaSdk) {
    if (!fs.existsSync(path.join(cudaSdk, "include/cuda.h")) || !fs.existsSync(path.join(cudaSdk, "include/nvml.h"))) {
      throw new Error("Static CUDA adapter needs the selected CUDA and NVML SDK headers");
    }
    Object.assign(definitions, {MOM_DPCPP_CUDA_SDK: cudaSdk});
  }
  if (jitObject) {
    const receipt = JSON.parse(fs.readFileSync(path.join(path.dirname(jitObject), "result.json"), "utf8"));
    const bytes = fs.readFileSync(jitObject);
    if (receipt.prefix !== "mom_dpcpp23_" || receipt.objectSha256 !== createHash("sha256").update(bytes).digest("hex")) {
      throw new Error("Source-JIT object does not match its isolated compiler receipt");
    }
    // Freeze the selected artifact before configure; an unrelated rebuild cannot replace it.
    const pinnedObject = path.join(output, "source-jit.o");
    fs.writeFileSync(pinnedObject, bytes, {flag: "wx"});
    Object.assign(definitions, {MOM_DPCPP_JIT_OBJECT: pinnedObject});
    fs.writeFileSync(path.join(output, "source-jit-input.json"), JSON.stringify(receipt, null, 2) + "\n");
  }
  if (adapter || cudaSdk) {
    // Upstream hwloc's autotools run in-source. Give it a private tracked-source snapshot so
    // generated configure/objects cannot mutate inputs or collide with another build.
    const snapshot = path.join(output, "hwloc-source"), archive = path.join(output, "hwloc-source.tar");
    fs.mkdirSync(snapshot);
    build("git", ["-C", /** @type {string} */ (hwloc), "archive", "--format=tar", "-o", archive, "HEAD"], "hwloc-snapshot");
    build("tar", ["-xf", archive, "-C", snapshot], "hwloc-unpack");
    Object.assign(definitions, {MOM_DPCPP_HWLOC: snapshot});
    if (adapter) {Object.assign(definitions, {MOM_DPCPP_LEVEL_ZERO_SOURCE: adapter.levelZero, MOM_DPCPP_COMPUTE_HEADERS: adapter.computeHeaders});}
  }
  const configure = ["-S", path.join(repo, "tests/native/dpcpp-static"), "-B", output, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    ...Object.entries(definitions).map(([key, value]) => `-D${key}=${value}`)];
  const compile = ["--build", output, "--target", "native-dpcpp-core", "--parallel", "4", "--", "-k", "0"];
  fs.writeFileSync(path.join(output, "commands.json"), JSON.stringify({configure, compile}, null, 2) + "\n");
  build("cmake", configure, "configure"); build("cmake", compile, "build");
  const receipt = auditBuild(output, Boolean(adapter), Boolean(jitObject), Boolean(cudaSdk), Boolean(opencl));
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({pins, ...receipt,
    adapter: [adapter ? "level_zero" : "", cudaSdk ? "cuda" : "", opencl ? "opencl" : ""].filter(Boolean).join(",") || "none", sourceJit: Boolean(jitObject), buildOnly: true,
    scope: "Whole-archive maintained DPC++ host core/UR with explicit optional static adapter/source-JIT. No execution/device/mining/performance qualification."}, null, 2) + "\n");
  console.log("Static DPC++ core built; NOT executed. Host-JIT/device/mining/performance execution remain separate gates.");
}

module.exports = {pins, sdkIdentity, adapterInputs, sourceJitInput, cudaSdkInput, openclInputs, auditBuild};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
