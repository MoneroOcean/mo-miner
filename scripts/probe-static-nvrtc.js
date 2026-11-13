"use strict";

// CPU-only compiler evidence. Never discover GPUs or run the generated code here.
const fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {fileHash} = require("./build-source-inputs");
const {nativeDependencies, auditNativeTrace} = require("./native-audit");

/** @typedef {{receiptSha256: string, files: Record<string, string>}} NvrtcSdkInputs */
/** @typedef {{dev: bigint, ino: bigint, size: bigint, mode: bigint, nlink: bigint, mtimeNs: bigint, ctimeNs: bigint}} NvrtcInputStat */

// The Linux/Windows entries remain the compiler-probed inputs. Linux 12.6 is a separate
// build-only image-worker cohort; CUDA runtime pins are separate. These archives are vendor
// prebuilts, and acquisition or a compiler probe does not establish redistribution rights.
const nvrtcPins = Object.freeze({
  linux: Object.freeze({version: "13.3.33", manifest: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json",
    archives: Object.freeze({
      nvrtc: Object.freeze({fileName: "cuda_nvrtc-linux-x86_64-13.3.33-archive.tar.xz",
        sha256: "9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a"}),
      nvptxcompiler: Object.freeze({fileName: "libnvptxcompiler-linux-x86_64-13.3.33-archive.tar.xz",
        sha256: "6c362578a8de750d28ae52efe5b3d2927983584293a9be24d40893e0f6ed73dd"}),
    })}),
  windows: Object.freeze({version: "13.1.115", manifest: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.1.1.json",
    archives: Object.freeze({
      nvrtc: Object.freeze({fileName: "cuda_nvrtc-windows-x86_64-13.1.115-archive.zip",
        sha256: "4ddd5a1e34fd62bb41e78c0725edfbf5609f4ecedd7fe118eddf2148b097fc91"}),
      nvptxcompiler: Object.freeze({fileName: "libnvptxcompiler-windows-x86_64-13.1.115-archive.zip",
        sha256: "bad2c1087563837d71c424950d2d896fd0db15fdc81a3fb2c483a4ce9450b0f4"}),
    })}),
  "linux-12.6": Object.freeze({version: "12.6.85", manifest: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json",
    archives: Object.freeze({nvrtc: Object.freeze({fileName: "cuda_nvrtc-linux-x86_64-12.6.85-archive.tar.xz",
      sha256: "a4289ba86f0deeccf4e1d94c58559eae4c8184fcb6eaade8fcab45584e6cf402"})}),
  }),
});

/** @typedef {"linux" | "windows"} Platform */
/** @typedef {"linux" | "windows" | "linux-12.6"} NvrtcCohort */
/** @typedef {{version: string, manifest: string, archives: Record<string, {fileName: string, sha256: string}>}} NvrtcPinSet */

/** Resolve an explicit compiler cohort without changing either platform's default.
 * @param {Platform} platform @param {NvrtcCohort} [cohort] @returns {NvrtcPinSet & {platform: Platform, cohort: NvrtcCohort}} */
function nvrtcCohortInfo(platform, cohort = platform) {
  if (platform !== "linux" && platform !== "windows") {throw new Error("NVRTC platform must be linux or windows");}
  if (cohort !== "linux" && cohort !== "windows" && cohort !== "linux-12.6") {throw new Error("NVRTC cohort is not available for the selected platform");}
  if ((cohort === "linux-12.6" && platform !== "linux") || (cohort !== "linux-12.6" && cohort !== platform)) {
    throw new Error("NVRTC cohort is not available for the selected platform");
  }
  return {platform, cohort, ...nvrtcPins[cohort]};
}

/** Keep an operation-local identity snapshot so replacing and restoring an input during
 * compiler/oracle execution cannot leave the same bytes as a false continuity proof.
 * SDK directory metadata also detects temporary replacement of the Linux SONAME aliases.
 * @param {Record<string, string>} files @param {string} sdk @returns {Record<string, NvrtcInputStat>} */
function snapshotNvrtcInputs(files, sdk) {
  const names = [...Object.keys(files), sdk, path.join(sdk, "lib64")].sort();
  if (!names.length || names.length > 64) {throw new Error("NVRTC SDK admission has an invalid input-file count");}
  return Object.fromEntries(names.map(file => {
    const stat = fs.lstatSync(file, {bigint: true});
    if ((!stat.isFile() && !stat.isDirectory()) || stat.isSymbolicLink()) {throw new Error("NVRTC SDK input is not a regular file/directory: " + file);}
    return [file, {dev: stat.dev, ino: stat.ino, size: stat.size, mode: stat.mode, nlink: stat.nlink,
      mtimeNs: stat.mtimeNs, ctimeNs: stat.ctimeNs}];
  }));
}

/** @param {Record<string, NvrtcInputStat>} expected */
function verifyNvrtcInputSnapshots(expected) {
  for (const [file, prior] of Object.entries(expected)) {
    const stat = fs.lstatSync(file, {bigint: true});
    if ((!stat.isFile() && !stat.isDirectory()) || stat.isSymbolicLink() || !isDeepStrictEqual(prior,
      {dev: stat.dev, ino: stat.ino, size: stat.size, mode: stat.mode, nlink: stat.nlink, mtimeNs: stat.mtimeNs, ctimeNs: stat.ctimeNs})) {
      throw new Error("NVRTC SDK input identity changed during compiler proof: " + file);
    }
  }
}

/** @param {string} sdkLib @param {NodeJS.ProcessEnv} [env] @returns {NodeJS.ProcessEnv} */
function oracleEnvironment(sdkLib, env = process.env) {
  // NVRTC dlopens its builtins itself, so that lookup cannot rely on the oracle's RUNPATH.
  // Drop empty components to avoid loader fallback to the current directory.
  const ldLibraryPath = [sdkLib, ...(env["LD_LIBRARY_PATH"] || "").split(path.delimiter)].filter(Boolean).join(path.delimiter);
  return {...env, LD_LIBRARY_PATH: ldLibraryPath};
}

/** @param {string} command @param {string[]} args @param {string} log @param {number} [status] @param {NodeJS.ProcessEnv} [env] */
function run(command, args, log, status = 0, env = process.env) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 120000, maxBuffer: 8 << 20, env});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== status) {throw new Error(`Static NVRTC proof failed: ${log}`, {cause: result.error});}
  return result.stdout;
}

function main() {
  const [sdkArg, outputArg] = process.argv.slice(2);
  if (process.platform !== "linux" || !sdkArg || process.argv.length > 4) {
    throw new Error("Usage (Linux): node scripts/probe-static-nvrtc.js <CUDA SDK> [evidence parent]");
  }
  const sdk = fs.realpathSync(sdkArg), repo = path.resolve(__dirname, "..");
  const vendorAssembler = fs.existsSync(path.join(sdk, "sdk.json")) ?
    /** @type {{nvrtcSdkInputs: (platform: "linux", root: string) => NvrtcSdkInputs}} */ (require("./prepare-native-nvrtc-sdk")) : undefined;
  // Admission precedes the first compiler/oracle invocation. Keep the file identities only for
  // this operation; the compact receipt hash is persisted and the full closure is re-admitted.
  const vendorBefore = vendorAssembler?.nvrtcSdkInputs("linux", sdk), vendorSnapshots = vendorBefore ? snapshotNvrtcInputs(vendorBefore.files, sdk) : undefined;
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-nvrtc-"));
  console.log(`Static NVRTC evidence: ${output}`);
  const archives = ["nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static"].map((name) => path.join(sdk, "lib64", `lib${name}.a`));
  for (const file of archives) {if (!fs.existsSync(file)) {throw new Error(`Missing matching compiler archive: ${file}`);}}
  const cxx = process.env["CXX"] || "c++";
  const flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-ffunction-sections", "-fdata-sections",
    "-isystem", path.join(sdk, "include"), path.join(repo, "tests/reference/native_nvrtc_static.cpp"),
    "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc"];
  const executable = path.join(output, "nvrtc-native"), dynamic = path.join(output, "nvrtc-oracle");
  run(cxx, [...flags, "-DMOM_STATIC_NVRTC", "-Wl,--start-group", ...archives, "-Wl,--end-group", "-ldl", "-lpthread", "-lm", "-o", executable], path.join(output, "build.log"));
  run(cxx, [...flags, "-L", path.join(sdk, "lib64"), `-Wl,-rpath,${path.join(sdk, "lib64")}`, "-lnvrtc", "-ldl", "-lpthread", "-lm", "-o", dynamic], path.join(output, "oracle-build.log"));
  const actualDir = path.join(output, "actual"), expectedDir = path.join(output, "expected");
  fs.mkdirSync(actualDir); fs.mkdirSync(expectedDir);
  const actual = run(executable, [actualDir], path.join(output, "compile.log"));
  const expected = run(dynamic, [expectedDir], path.join(output, "oracle.log"), 0, oracleEnvironment(path.join(sdk, "lib64")));
  if (actual !== expected || !actual.includes("GPU dispatches=0")) {throw new Error("Static/shared compiler transcripts differ");}
  const files = fs.readdirSync(actualDir).map((name) => {
    const data = fs.readFileSync(path.join(actualDir, name)), oracle = fs.readFileSync(path.join(expectedDir, name));
    if (!data.equals(oracle)) {throw new Error(`Static/shared GPU code differs: ${name}`);}
    return {name, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex")};
  });
  if (files.length !== 4) {throw new Error("Incomplete compiler target/output matrix");}
  // Exercise the same linker family required by real AdaptiveCpp ThinLTO archives. NVRTC's
  // legacy .ctors otherwise link successfully but leave the compiler's architecture list empty.
  const tools = run(process.env["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config", ["--bindir"], path.join(output, "llvm-tools.log")).trim();
  const lld = path.join(output, "nvrtc-lld"), lldDir = path.join(output, "lld");
  fs.mkdirSync(lldDir);
  run(path.join(tools, "clang++"), [`--ld-path=${path.join(tools, "ld.lld")}`, ...flags, "-DMOM_STATIC_NVRTC",
    "-Wl,-T," + path.join(repo, "runtime/static-constructors.ld"), "-Wl,--start-group", ...archives, "-Wl,--end-group",
    "-ldl", "-lpthread", "-lm", "-o", lld], path.join(output, "lld-build.log"));
  if (run(lld, [lldDir], path.join(output, "lld.log")) !== expected || files.some(({name}) =>
    !fs.readFileSync(path.join(lldDir, name)).equals(fs.readFileSync(path.join(expectedDir, name))))) {
    throw new Error("Static LLD/compiler initialization or output differs from the shared oracle");
  }
  run(executable, [], path.join(output, "arguments.log"), 2);
  const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")));
  const trace = path.join(output, "syscalls.log");
  run("strace", ["-f", "-s", "1024", "-e", "trace=openat,execve,execveat", "-E", "PATH=/unavailable", "-E", "LD_LIBRARY_PATH=/unavailable", "-o", trace, executable, actualDir], path.join(output, "isolated-paths.log"));
  // NVRTC may optionally probe the system driver even during CPU-only compilation. This never
  // admits application compiler libraries; a separate no-device/no-driver container must pass.
  auditNativeTrace(fs.readFileSync(trace, "utf8"), executable, 1, {cudaDriver: true});
  const hash = (/** @type {string} */ file) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  /** @type {{receiptSha256: string} | undefined} */ let vendor;
  if (vendorBefore && vendorAssembler && vendorSnapshots) {
    const vendorAfter = vendorAssembler.nvrtcSdkInputs("linux", sdk);
    if (vendorAfter.receiptSha256 !== vendorBefore.receiptSha256 || !isDeepStrictEqual(vendorAfter.files, vendorBefore.files)) {
      throw new Error("NVRTC SDK receipt or admitted inputs changed during compiler proof");
    }
    verifyNvrtcInputSnapshots(vendorSnapshots);
    vendor = {receiptSha256: vendorAfter.receiptSha256};
  }
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({sdk, bytes: fs.statSync(executable).size, needed,
    archives: archives.map((file) => ({file, sha256: fileHash(file)})), files,
    source: {file: "sycl/cn_gpu/cuda_expand_kernel.inc", sha256: hash(path.join(repo, "sycl/cn_gpu/cuda_expand_kernel.inc"))},
    lld: {bytes: fs.statSync(lld).size, constructorScript: hash(path.join(repo, "runtime/static-constructors.ld")), byteExactOracle: true},
    ...(vendor ? {vendor} : {}),
    scope: "Maintained CN/GPU source compiled statically, byte-exact shared compiler oracle; CPU only, no GPU/runtime/performance qualification"}, null, 2) + "\n");
  console.log("Static NVRTC matches shared compiler; no application runtime/compiler dependency. GPU execution remains unqualified.");
}

// The preparer imports this pin table. Assign the export before CLI main() can lazily load it.
module.exports = {nvrtcPins, nvrtcCohortInfo, oracleEnvironment};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
