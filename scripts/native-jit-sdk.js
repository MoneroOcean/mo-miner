"use strict";

// Shared original compiler graph for ELF and COFF SDK production. Planning is side-effect
// free; a plan alone is not proof that sources, resources or compiled archives are qualified.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {sdkComponents} = require("./prepare-native-sdk-sources");
const {fileHash, sourceInputs, verifySourceInputs, buildFileInputs, completeBuildInputs, readBuildJson} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const repo = path.resolve(__dirname, "..");
// One footprint for source emission and resource admission; no separate generated algorithm.
const jitSourceInputs = Object.freeze(["tests/reference/native-dpcpp-jit.cpp", "tests/reference/native-kawpow-source.cpp",
  "tests/native/dpcpp-jit/CMakeLists.txt", "scripts/native-resources.js", "sycl/kawpow/device.inc",
  "sycl/kawpow/epoch.h", "sycl/kawpow/jit.inc", "sycl/kawpow/keccak.inc", "sycl/kawpow/program.inc",
  "sycl/device-state.h", "sycl/lib-internal.h", "sycl/lib.h", "sycl/workgroup-limits.h",
  "xmrig/3rdparty/libethash/data_sizes.h", "xmrig/3rdparty/libethash/ethash.h"]);
/** CMake values are literals, never user-supplied lists, expressions or flags.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
  path.normalize(value) === value && !/[;"$\\]/.test(value) && !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);
/** @typedef {{cmake: string, ninja: string, cc?: string, cxx?: string, toolchain?: string}} Tools */

/** Preserve the original target dependency order, static compiler and in-process frontend.
 * The proof target supplies upstream's actual transitive archive closure to the existing
 * isolation code; do not maintain a second list of LLVM libraries here.
 * @param {'linux'|'windows'} target @param {Record<string,string>} roots @param {string} sdk
 * @param {string} resources @param {string} output @param {Tools} tools @param {number} jobs */
function jitSdkPlan(target, roots, sdk, resources, output, tools, jobs) {
  const toolKeys = tools && Object.keys(tools).sort().join(",");
  if (!["linux", "windows"].includes(target) || !roots || Object.keys(roots).sort().join(",") !== Object.keys(sdkComponents("compiler")).sort().join(",") ||
      !["cmake,ninja", "cmake,ninja,toolchain", "cc,cmake,cxx,ninja", "cc,cmake,cxx,ninja,toolchain"].includes(toolKeys || "") ||
      target === "windows" && !tools.toolchain || ![...Object.values(roots), sdk, resources, output, ...Object.values(tools)].every(literal) ||
      !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {throw new Error("Static JIT SDK needs a supported target, complete original roots, literal tools/resources and jobs in1..16");}
  const nested = (/** @type {string} */ a, /** @type {string} */ b) => {const p = path.relative(a, b); return !p || !path.isAbsolute(p) && p !== ".." && !p.startsWith(".." + path.sep);};
  for (const input of [...Object.values(roots), sdk, resources, repo, tools.cmake, tools.ninja,
    ...(tools.cc && tools.cxx ? [tools.cc, tools.cxx] : []), ...(tools.toolchain ? [tools.toolchain] : [])]) {
    if (nested(input, output) || nested(output, input)) {throw new Error("Static JIT SDK output overlaps an input");}
  }
  const build = path.join(output, "build"), projects = ["sycl", "llvm-spirv", "libdevice", "sycl-jit"];
  const embeddedJit23C = path.join(sdk, "bin", target === "windows" ? "clang-cl" : "clang");
  const embeddedJit23Cxx = path.join(sdk, "bin", target === "windows" ? "clang-cl" : "clang++");
  const hostC = tools.cc || embeddedJit23C, hostCxx = tools.cxx || embeddedJit23Cxx;
  const compilerIdentity = {host: {mode: tools.cc ? "explicit-host-override" : "matching-clang23-bootstrap", cc: hostC, cxx: hostCxx},
    embeddedJit23: embeddedJit23Cxx};
  /** @type {Record<string,string>} */
  const definitions = {CMAKE_BUILD_TYPE: "Release", CMAKE_C_COMPILER: hostC,
    CMAKE_CXX_COMPILER: hostCxx, CMAKE_MAKE_PROGRAM: tools.ninja,
    BUILD_SHARED_LIBS: "OFF", LLVM_BUILD_LLVM_DYLIB: "OFF", LLVM_BUILD_LLVM_C_DYLIB: "OFF", CLANG_LINK_CLANG_DYLIB: "OFF",
    // LLVM_ENABLE_PROJECTS requires upstream sibling directories, even with explicit paths.
    // Our proof is out-of-tree: only the external-project registry may enable it.
    LIBCLANG_BUILD_STATIC: "ON", LLVM_TARGETS_TO_BUILD: "X86;NVPTX;SPIRV", LLVM_ENABLE_PROJECTS: ["clang", ...projects].join(";"),
    LLVM_EXTERNAL_PROJECTS: [...projects, "mom-jit-proof"].join(";"), LLVM_ENABLE_EH: "ON", LLVM_ENABLE_RTTI: "ON", LLVM_ENABLE_PIC: "ON",
    LLVM_APPEND_VC_REV: "OFF", CLANG_SPAWN_CC1: "OFF", SYCL_HEADERS_ONLY: "ON", SYCL_ENABLE_WERROR: "OFF",
    SYCL_ENABLE_XPTI_TRACING: "OFF", SYCL_JIT_STATIC: "ON", SYCL_JIT_RESOURCE_OVERLAY: resources,
    "FETCHCONTENT_SOURCE_DIR_VC-INTRINSICS": roots["vcIntrinsics"] || "", CMAKE_DISABLE_FIND_PACKAGE_LLVMGenXIntrinsics: "TRUE",
    LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR: roots["spirvHeaders"] || "", FETCHCONTENT_FULLY_DISCONNECTED: "ON", CMAKE_POLICY_DEFAULT_CMP0170: "NEW",
    LLVM_EXTERNAL_MOM_JIT_PROOF_SOURCE_DIR: path.join(repo, "tests/native/dpcpp-jit"), MOM_SOURCE_REPO: repo};
  // Windows resource generation stays on the original SDK Clang23 tool, independently of
  // the selected CMake host compiler. The resource compiler is not this C/C++ override.
  if (target === "windows") {definitions["SYCL_JIT_RESOURCE_CLANG"] = embeddedJit23C;}
  for (const name of ["ASSERTIONS", "PLUGINS", "BINDINGS", "CURL", "HTTPLIB", "LIBEDIT", "LIBPFM", "LIBXML2", "ZLIB", "ZSTD"]) {definitions["LLVM_ENABLE_" + name] = "OFF";}
  for (const name of ["UTILS", "TESTS", "DOCS", "BENCHMARKS", "EXAMPLES"]) {definitions["LLVM_INCLUDE_" + name] = "OFF";}
  for (const name of ["TESTS", "DOCS"]) {definitions["CLANG_INCLUDE_" + name] = "OFF";}
  for (const name of ["HLSL", "OBJC_REWRITER", "LIBXML2"]) {definitions["CLANG_ENABLE_" + name] = "OFF";}
  definitions["CLANG_BUILD_EXAMPLES"] = "OFF";
  for (const name of projects) {definitions["LLVM_EXTERNAL_" + name.toUpperCase().replaceAll("-", "_") + "_SOURCE_DIR"] = path.join(roots["dpcpp"] || "", name);}
  if (tools.toolchain) {definitions["CMAKE_TOOLCHAIN_FILE"] = tools.toolchain;}
  return {target, build, definitions, compilerIdentity,
    configure: [tools.cmake, "-S", path.join(roots["dpcpp"] || "", "llvm"), "-B", build, "-G", "Ninja", ...Object.entries(definitions).map(([key, value]) => "-D" + key + "=" + value)],
    compile: [tools.cmake, "--build", build, "--target", "native-dpcpp-jit", "--parallel", String(jobs), "--", "-k", "0"]};
}

/** Bootstrap resource discovery with the maintained CPU source generator, not a linked JIT.
 * SDK host headers supply type declarations only; no SYCL runtime or GPU driver is linked.
 * @param {string} sdk @param {string} output */
function jitEmitterPlan(sdk, output) {
  if (![sdk, output].every(literal)) {throw new Error("JIT source emitter needs literal absolute SDK/output paths");}
  const nested = (/** @type {string} */ a, /** @type {string} */ b) => {const p = path.relative(a, b); return !p || !path.isAbsolute(p) && p !== ".." && !p.startsWith(".." + path.sep);};
  for (const input of [sdk, repo]) {if (nested(input, output) || nested(output, input)) {throw new Error("JIT source emitter output overlaps an input");}}
  const cxx = path.join(sdk, "bin/clang++"), binary = path.join(output, "native-kawpow-source"), depfile = path.join(output, "emitter.d");
  return {cxx, binary, depfile,
    resources: [process.execPath, path.join(repo, "scripts/native-resources.js"), path.join(output, "kawpow-jit-resources.inc"),
      "kawpow-device.inc", path.join(repo, "sycl/kawpow/device.inc"), "kawpow-keccak.inc", path.join(repo, "sycl/kawpow/keccak.inc")],
    compile: [cxx, "-std=c++17", "-O2", "-DMOM_EMBEDDED_SYCL_JIT", "-DMOM_SYCL_HAS_CUDA", "-DMOM_KAWPOW_SOURCE_EMITTER",
      "-I" + path.join(sdk, "include"), "-I" + output, "-MD", "-MF", depfile, path.join(repo, "tests/reference/native-kawpow-source.cpp"), "-o", binary]};
}

/** Compile only a new host emitter, retaining exact tool/source/header dependencies. Its
 * receipt feeds the existing resource collector but does not certify a compiler SDK.
 * No miner, emitted source, driver, compiler JIT or GPU is executed here.
 * Explicit target headers are also frozen before compilation: their ABI may differ from
 * the host emitter's own headers (notably Windows GNU on a Linux build host).
 * @param {string} sdk @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {unknown} [targetHeaders] */
function buildJitEmitter(sdk, output, environment = process.env, targetHeaders = []) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("JIT emitter production requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  jitEmitterPlan(sdk, output);
  if (!Array.isArray(targetHeaders) || targetHeaders.length > 16384 || !targetHeaders.every(literal)) {throw new Error("JIT emitter target headers must be bounded literal absolute files");}
  if (fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("Select a new JIT emitter output; existing evidence is never overwritten");}
  const selectedSdk = fs.realpathSync(sdk), selectedOutput = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  const plan = jitEmitterPlan(selectedSdk, selectedOutput), env = nativeSdkEnvironment(environment);
  const header = path.join(selectedSdk, "include/sycl/sycl.hpp");
  if (!fs.statSync(header, {throwIfNoEntry: false})?.isFile()) {throw new Error("JIT emitter requires the selected SDK's SYCL headers");}
  const leaves = [...new Set([...jitSourceInputs, "scripts/native-jit-sdk.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"])];
  const recipe = sourceInputs(repo, [], leaves);
  const initial = buildFileInputs([process.execPath, plan.cxx, header, ...targetHeaders, ...leaves.map(name => path.join(repo, name))].map(file => fs.realpathSync(file)));
  fs.mkdirSync(selectedOutput, {mode: 0o700});
  fs.writeFileSync(path.join(selectedOutput, "before.json"), JSON.stringify({plan, recipe, inputs: initial}, null, 2) + "\n", {flag: "wx"});
  const startedAt = Date.now();
  for (const [index, command] of [plan.resources, plan.compile].entries()) {
    const fd = fs.openSync(path.join(selectedOutput, index ? "compile.log" : "resources.log"), "wx");
    try {
      const result = spawnSync(command[0] || "", command.slice(1), {env, cwd: repo, stdio: ["ignore", fd, fd], timeout: 300000});
      if (result.status !== 0 || result.error || result.signal) {throw new Error("JIT source emitter stage failed; inspect " + selectedOutput, {cause: result.error});}
    } finally {fs.closeSync(fd);}
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], leaves), "JIT source emitter");
  const inputs = completeBuildInputs([plan.depfile], repo, selectedOutput, startedAt, initial);
  const receipt = {version: 1, fixture: plan.binary, fixtureSha256: fileHash(plan.binary), compiler: plan.cxx, compilerSha256: fileHash(plan.cxx),
    inputs, recipe, targetHeaders: targetHeaders.map(file => fs.realpathSync(file)), buildOnly: true, executed: false, publicationQualified: false,
    scope: "Maintained CPU-only source generator, no linked compiler SDK or SYCL runtime. Not compiler SDK/runtime/publication qualification."};
  fs.writeFileSync(path.join(selectedOutput, "result.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built original CPU-only source emitter; no compiler SDK or GPU execution.");
  return receipt;
}
module.exports = {jitSdkPlan, jitEmitterPlan, buildJitEmitter, jitSourceInputs, literal};
if (require.main === module) {
  try {
    const [mode, sdk, output, headers, extra] = process.argv.slice(2);
    if (mode !== "emitter" || !sdk || !output || extra || headers !== undefined && !literal(headers)) {throw new Error("Usage: native-jit-sdk.js emitter /absolute/DPC++ /new/absolute/output [/absolute/target-headers.json]");}
    buildJitEmitter(sdk, output, process.env, headers === undefined ? [] : readBuildJson(headers));
  } catch (error) {console.error(error instanceof Error ? error.message : "JIT source emitter production failed"); process.exitCode = 1;}
}
