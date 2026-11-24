"use strict";

// Produce the original Linux SYCL/UR/driver-loader SDK from the same target-scoped source
// materializer used on Windows. Reuse the CMake graph and whole-archive audit, never run it.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {verifyPreparedSdkSources, sdkSourceFiles, validateSourceRoots} = require("./prepare-native-sdk-sources");
const {pins, sdkIdentity, auditBuild} = require("./native-dpcpp-pins");
const {hasExactBuildFields, buildPathsOverlap, fileHash, readBuildJson, sourceInputs, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {compilerEnvironment: sdkCompilerEnvironment, sdkHostEnvironment, verifyCompilerEnvironment} = require("./build-native-sycl-sdk");
const {isStaticArchive} = require("./native-audit");
const {isDeepStrictEqual} = require("node:util"), {dpcppEntries} = require("./native-compiler-isolation");
const {exportDpcppComponent} = require("./native-dpcpp-component");
const {requireLinuxSysroot, linuxSysrootRecord} = require("./native-linux-sysroot");
const repo = path.resolve(__dirname, "..");

/** Share the producer's exact source graph with warm SDK admission instead of copying its file list. */
function linuxStaticSdkRecipe() {
  return sourceInputs(repo, ["tests/native/dpcpp-static"], ["scripts/build-native-dpcpp-sdk.js", "scripts/prepare-native-sdk-sources.js",
    "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/build-native-sycl-sdk.js", "scripts/build-native-release.js",
    "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-jit-link.js", "scripts/native-comgr-inputs.js", "scripts/native-dpcpp-pins.js",
    "scripts/native-audit.js", "scripts/native-dpcpp-component.js", "scripts/native-linux-sysroot.js",
    "scripts/native-cuda-adapter.js", "scripts/native-level-zero-adapter.js", "scripts/native-compiler-isolation.js", "runtime/system-driver.h",
    "runtime/driver-library.cpp", "tests/reference/native-dpcpp-static.cpp", "tests/reference/native-static-adapter-selection.cpp",
    "tests/reference/native-umf-static.cpp", "tests/reference/native-level-zero-static.cpp", "tests/reference/native-opencl-binding.cpp", "tests/native/driver-library.cpp"]);
}
/** CMake tokens must be absolute literals, not list/interpolation fragments.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && !/[;"$\\]/.test(value) &&
  !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
/** @typedef {{cmake: string, ninja: string, cc?: string, cxx?: string}} Tools */

/** Full Linux runtime closure, not a different algorithm implementation. Source-JIT is
 * mandatory here: a successful reduced SDK cannot silently disable existing mining JIT.
 * @param {Record<string, string>} roots @param {string} sdk @param {string} cuda
 * @param {string} jit @param {string} output @param {Tools} tools @param {number} jobs */
function linuxStaticSdkPlan(roots, sdk, cuda, jit, output, tools, jobs) {
  const toolKeys = ["cmake", "ninja"];
  if (!roots || !hasExactBuildFields(roots, ["computeHeaders", "dpcpp", "emhash", "hwloc", "levelZero", "openclHeaders", "openclLoader", "umf"]) ||
      (!hasExactBuildFields(tools, toolKeys) && !hasExactBuildFields(tools, [...toolKeys, "cc", "cxx"])) || ![...Object.values(roots), sdk, cuda, jit, output, ...Object.values(tools)].every(literal) ||
      !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {throw new Error("Linux static SDK needs all original roots, explicit tools/JIT and jobs in1..16");}
  for (const input of [...Object.values(roots), sdk, cuda, path.dirname(jit), repo, ...Object.values(tools)]) {
    if (buildPathsOverlap(input, output)) {throw new Error("Linux static SDK output overlaps an input");}
  }
  const build = path.join(output, "build"), hostCompilers = {cc: tools.cc || path.join(sdk, "bin/clang"), cxx: tools.cxx || path.join(sdk, "bin/clang++")};
  // Autoconf's deferred $CC expansion needs safe command names, not shell quotes in env values.
  const compilerEnvironment = sdkCompilerEnvironment(hostCompilers);
  const definitions = {MOM_DPCPP_SOURCE: roots["dpcpp"], MOM_DPCPP_SDK: sdk, MOM_DPCPP_EMHASH: roots["emhash"],
    MOM_DPCPP_UMF: roots["umf"], MOM_DPCPP_LEVEL_ZERO_SOURCE: roots["levelZero"], MOM_DPCPP_COMPUTE_HEADERS: roots["computeHeaders"],
    MOM_DPCPP_HWLOC: path.join(output, "hwloc"), MOM_DPCPP_OPENCL_SOURCE: roots["openclLoader"], MOM_DPCPP_OPENCL_HEADERS: roots["openclHeaders"],
    MOM_DPCPP_CUDA_SDK: cuda, MOM_DPCPP_JIT_OBJECT: jit, MOM_SOURCE_REPO: repo,
    CMAKE_CXX_COMPILER: hostCompilers.cxx, CMAKE_C_COMPILER: hostCompilers.cc, CMAKE_MAKE_PROGRAM: tools.ninja};
  return {build, hostCompilers, compilerEnvironment, configure: [tools.cmake, "-S", path.join(repo, "tests/native/dpcpp-static"), "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    ...Object.entries(definitions).map(([name, value]) => "-D" + name + "=" + value)],
  compile: [tools.cmake, "--build", build, "--target", "native-dpcpp-core", "--parallel", String(jobs), "--", "-k", "0"]};
}

/** Isolated source-JIT objects retain their existing original compiler's explicit receipt.
 * A receipt is local provenance continuity, not a signature for arbitrary downloaded code.
 * @param {string} object */
function linuxSdkJitInput(object) {
  if (!literal(object) || !fs.lstatSync(object).isFile()) {throw new Error("Select a regular isolated Linux source-JIT object");}
  const file = path.join(path.dirname(object), "result.json"), receipt = readBuildJson(file), sha256 = fileHash(object);
  if (receipt.prefix !== "mom_dpcpp23_" || receipt.objectSha256 !== sha256 || !isDeepStrictEqual(receipt.entryPoints, dpcppEntries)) {
    throw new Error("Linux source-JIT object differs from its original isolation receipt");
  }
  return {[object]: sha256, [file]: fileHash(file)};
}

/** Catch incomplete compiler/driver SDK installs before configuring or mutating output.
 * NVML headers are separate from CUDA on some installations; no driver SO is linked here.
 * @param {string} sdk @param {string} cuda */
function linuxSdkHeaders(sdk, cuda) {
  if (![sdk, cuda].every(literal)) {throw new Error("Select absolute Linux compiler/CUDA SDK roots");}
  for (const [root, names] of /** @type {[string, string[]][]} */ ([[sdk, ["sycl/sycl.hpp"]], [cuda, ["cuda.h", "nvml.h"]]])) {
    for (const name of names) {
      const file = path.join(root, "include", name), stat = fs.lstatSync(file, {throwIfNoEntry: false});
      if (!stat?.isFile() || stat.size < 1) {throw new Error("Linux static SDK requires a nonempty original header: " + file);}
    }
  }
  return [path.join(sdk, "include"), path.join(cuda, "include")].map(root => ({root, files: sdkSourceFiles(root)}));
}

/** Parse only Ninja's valid completed dependency records. Unknown/stale output fails closed;
 * paths may contain spaces, but no controls or unbounded metadata. @param {string} text @param {string} base */
function linuxSdkDependencies(text, base) {
  if (Buffer.byteLength(text) > (16 << 20) || !literal(base)) {throw new Error("Invalid bounded Ninja SDK dependencies");}
  const files = new Set(); let record = false;
  for (const line of text.split(/\r?\n/)) {
    if (!line) {record = false; continue;}
    if (/^.+: #deps [0-9]+, deps mtime [0-9]+ \(VALID\)$/.test(line)) {record = true; continue;}
    if (!record || !line.startsWith("    ") || !line.slice(4) || Array.from(line).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
      throw new Error("Unrecognized or stale Ninja SDK dependency record");
    }
    files.add(path.resolve(base, line.slice(4))); if (files.size > 65536) {throw new Error("Excessive Ninja SDK dependencies");}
  }
  if (!files.size) {throw new Error("Missing Ninja SDK dependencies");}
  return [...files];
}

/** Build and inspect only. Frozen original sources and SDK headers are rechecked after
 * all independent compiler work; failed stages retain their logs without issuing success.
 * @param {string} sourcesFile @param {string} sdk @param {string} cuda @param {string} jit
 * @param {string} output @param {Tools} tools @param {number} jobs @param {NodeJS.ProcessEnv} [environment] */
function buildLinuxStaticSdk(sourcesFile, sdk, cuda, jit, output, tools, jobs, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Linux SDK production requires Linuxx64 and MOM_AOT_BUILD_ONLY=1");}
  const env = nativeSdkEnvironment(environment), sysroot = requireLinuxSysroot(env);
  const sourceHash = fileHash(sourcesFile), selected = verifyPreparedSdkSources(sourcesFile, "linux");
  const roots = validateSourceRoots(Object.fromEntries(Object.entries(selected.prepared).map(([name, row]) => [name, row.root])), output);
  const hwloc = roots["hwloc"]; if (!hwloc) {throw new Error("Missing admitted hwloc source root");}
  if (path.join(fs.realpathSync(path.dirname(output)), path.basename(output)) !== output) {throw new Error("Select a canonical Linux SDK output");}
  // Reject overlap through symlinked SDK/output parents too, before any build mutation.
  linuxStaticSdkPlan(roots, fs.realpathSync(sdk), fs.realpathSync(cuda), fs.realpathSync(jit),
    path.join(fs.realpathSync(path.dirname(output)), path.basename(output)), tools, jobs);
  const plan = linuxStaticSdkPlan(roots, sdk, cuda, jit, output, tools, jobs), buildEnv = sdkHostEnvironment(env, plan.hostCompilers);
  verifyCompilerEnvironment(plan.hostCompilers, buildEnv);
  if ([output, output + ".component"].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {throw new Error("Linux SDK output and installed component must be new");}
  const recipe = linuxStaticSdkRecipe();
  const inputs = {...linuxSdkJitInput(jit), [sourcesFile]: sourceHash,
    ...Object.fromEntries([...Object.values(tools), path.join(sdk, "bin/clang"), path.join(sdk, "bin/clang++"), path.join(sdk, ".mom-toolchain-sha256")].map(file => [file, fileHash(file)]))};
  const headers = linuxSdkHeaders(sdk, cuda);
  // The host compiler may change without changing the pinned SYCL headers/JIT ABI.
  const compilerVersions = /** @type {Record<string,string>} */ ({});
  for (const compiler of new Set([path.join(sdk, "bin/clang++"), ...Object.values(plan.hostCompilers)])) {
    const probe = spawnSync(compiler, ["--version"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    if (probe.status !== 0 || probe.error || probe.signal || !probe.stdout?.trim() || probe.stdout.includes("\0")) {throw new Error("Linux SDK compiler identity query failed: " + compiler);}
    compilerVersions[compiler] = probe.stdout;
  }
  sdkIdentity(compilerVersions[path.join(sdk, "bin/clang++")] || "", fs.readFileSync(path.join(sdk, ".mom-toolchain-sha256"), "utf8"));
  fs.mkdirSync(output, {mode: 0o700});
  // hwloc's original autotools build writes in-source. Its own private snapshot may change;
  // the admitted, shared source-preparation cohort stays immutable for both targets.
  fs.cpSync(hwloc, path.join(output, "hwloc"), {recursive: true, errorOnExist: true, force: false});
  fs.writeFileSync(path.join(output, "before.json"), JSON.stringify({sourcesFile, inputs, headers, recipe, hostCompilers: plan.hostCompilers,
    compilerEnvironment: plan.compilerEnvironment, compilerVersions, configure: plan.configure, compile: plan.compile}, null, 2) + "\n", {flag: "wx"});
  /** @param {string[]} command @param {string} label @param {boolean} [capture] @param {NodeJS.ProcessEnv} [commandEnv] */
  const run = (command, label, capture = false, commandEnv = env) => {
    const program = command[0]; if (!program) {throw new Error("Missing Linux SDK command");}
    const fd = fs.openSync(path.join(output, label + ".log"), "wx");
    try {
      const result = spawnSync(program, command.slice(1), {cwd: repo, env: commandEnv, encoding: "utf8", stdio: capture ? ["ignore", "pipe", fd] : ["ignore", fd, fd], timeout: 30 * 60 * 1000, maxBuffer: 16 << 20});
      if (result.status !== 0 || result.error || result.signal) {throw new Error("Linux SDK stage failed: " + label + "; inspect " + output);}
      return result.stdout || "";
    } finally {fs.closeSync(fd);}
  };
  console.log("[native-sdk] Building original Linux SYCL/UR and all static driver adapters");
  run(plan.configure, "configure"); run(plan.compile, "build", false, buildEnv);
  const dependencies = buildFileInputs(linuxSdkDependencies(run([tools.ninja, "-C", plan.build, "-t", "deps"], "dependencies", true, buildEnv), plan.build));
  verifyPreparedSdkSources(sourcesFile, "linux");
  verifySourceInputs(recipe, linuxStaticSdkRecipe(), "Linux SDK producer");
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("Linux SDK input changed during compilation: " + file);}}
  for (const header of headers) {verifySourceInputs({files: header.files, graph: {}}, {files: sdkSourceFiles(header.root), graph: {}}, "Linux SDK headers");}
  const archive = path.join(plan.build, "libmom-sycl-static.a");
  if (!isStaticArchive(archive)) {throw new Error("Linux SDK host runtime is not a static archive");}
  const audit = auditBuild(plan.build, true, true, true, true);
  const result = {version: 1, target: "linux", pins, sourcesFile, inputs, headers, dependencies, recipe, hostCompilers: plan.hostCompilers, compilerVersions,
    archive, sha256: fileHash(archive), audit, sysroot: linuxSysrootRecord(sysroot), buildOnly: true, executed: false, publicationQualified: false,
    scope: "Original full Linux SYCL/UR/static loader sources with isolated source JIT; no GPU, runtime, mining or publication qualification"};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  exportDpcppComponent(path.join(output, "result.json"), tools.ninja, output + ".component", environment);
  console.log("[native-sdk] Built original Linux static SYCL SDK; inspected, not executed.");
  return result;
}

module.exports = {linuxStaticSdkRecipe, linuxStaticSdkPlan, linuxSdkJitInput, linuxSdkHeaders, linuxSdkDependencies, buildLinuxStaticSdk};
if (require.main === module) {
  try {
    const [sources, sdk, cuda, jit, output, cmake, ninja, jobs = "2", cc, cxx, extra] = process.argv.slice(2);
    if (!sources || !sdk || !cuda || !jit || !output || !cmake || !ninja || extra || Boolean(cc) !== Boolean(cxx)) {
      throw new Error("Usage: build-native-dpcpp-sdk.js /sources.json /DPC++ /CUDA /isolated/compiler.o /new-output /cmake /ninja [jobs [host-cc host-cxx]]");
    }
    buildLinuxStaticSdk(sources, sdk, cuda, jit, output, {cmake, ninja, ...(cc && cxx ? {cc, cxx} : {})}, Number(jobs));
  } catch (error) {console.error(error instanceof Error ? error.message : "Linux SDK production failed"); process.exitCode = 1;}
}
