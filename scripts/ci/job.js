"use strict";

// Prepares one native CI job (.github/workflows/native-job.yml): exports the run-scoped roots the native
// scripts derive and verify, the caller's producer pins, and the exact artifact transfer plan. Artifact
// lists name files with "@" standing for "<run>-<attempt>", the suffix every producer embeds.
const fs = require("node:fs"), path = require("node:path");

/** Environment name suffix -> /tmp/mom-native-<stem>-<run>-<attempt>. */
const roots = Object.freeze({
  SOURCE: "prerequisites", LINUX_SDK: "llvm21-sdk", IO_SOURCE: "io-prerequisites", IO_SDK: "io-openssl-sdk",
  HERMES_SOURCE: "hermes-prerequisites", HERMES_SDK: "hermes-sdk", ACPP_SOURCE: "acpp-prerequisites",
  LINUX_SOURCES: "linux-source-prerequisites", DPCPP_CUDA_INPUTS: "dpcpp-cuda-inputs", GNU_HEADERS: "gnu-headers",
  WINDOWS_GNU_HEADERS: "windows-gnu-headers", LINUX_JIT_RESOURCES: "linux-jit-resources", LINUX_JIT_SDK: "linux-jit-sdk",
  ROCM_SOURCE: "rocm-prerequisites", ROCM_COMGR_SDK: "rocm-comgr-sdk", ROCM_RESOURCES: "rocm-resources",
  ROCM_HIP_HEADERS: "rocm-hip-headers", ROCM_HOST_SDK: "rocm-host-sdk", ZSTD_SOURCE: "zstd-prerequisites", ZSTD_SDK: "zstd-sdk",
  OPENMP_SDK: "openmp-sdk", ACPP_CUDA_HIP_SDK: "acpp-cuda-hip-sdk", NVRTC_LINUX_SOURCE: "nvrtc-linux-prerequisites",
  NVRTC_LINUX_SDK: "nvrtc-linux-sdk", NVRTC_LINUX_126_SOURCE: "nvrtc-linux-12.6-prerequisites",
  NVRTC_LINUX_126_SDK: "nvrtc-linux-12.6-sdk", CUDA_IMAGES: "cuda-images", LINUX_SDK_DESCRIPTOR: "linux-sdk-descriptor",
  NVRTC_WINDOWS_SOURCE: "nvrtc-windows-prerequisites", NVRTC_WINDOWS_SDK: "nvrtc-windows-sdk",
  WINDOWS_LLVM: "windows-llvm21", WINDOWS_JIT: "windows-jit", WINDOWS_JIT_RESOURCES: "windows-jit-resources",
});
const maxUploads = 4;

/** @param {NodeJS.ProcessEnv} environment */
function runSuffix(environment) {
  const run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (!/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {throw new Error("Native CI job needs the GitHub run identity");}
  return run + "-" + attempt;
}

/** Derive the run-scoped roots; a GPU factory job also gets its profile-specific root.
 * @param {NodeJS.ProcessEnv} environment @returns {Record<string, string>} */
function nativeCiEnvironment(environment) {
  const suffix = runSuffix(environment), profile = environment["MOM_NATIVE_WINDOWS_GPU_PROFILE"] || "";
  if (!/^[a-z-]{0,32}$/.test(profile)) {throw new Error("Native CI GPU profile is malformed");}
  const values = Object.fromEntries(Object.entries(roots).map(([name, stem]) =>
    ["MOM_NATIVE_" + name + "_DIR", "/tmp/mom-native-" + stem + "-" + suffix]));
  return profile ? {...values, MOM_NATIVE_WINDOWS_GPU_DIR: `/tmp/mom-native-windows-gpu-${profile}-${suffix}`} : values;
}

/** Parse caller pins: one KEY=value per line; values come from producer job outputs and must be present.
 * @param {string} text */
function parsePins(text) {
  /** @type {Record<string, string>} */
  const pins = {};
  for (const line of text.split("\n").map(row => row.trim()).filter(Boolean)) {
    const match = /^(MOM_[A-Z0-9_]+)=(\S.*)$/.exec(line), key = match?.[1] || "";
    if (!match || Object.hasOwn(pins, key)) {throw new Error("Native CI pin line is malformed, empty or repeated: " + line.split("=")[0]);}
    pins[key] = match[2] || "";
  }
  return pins;
}

/** Split a whitespace-separated artifact list and substitute the run suffix.
 * @param {string} text @param {string} suffix */
function names(text, suffix) {
  return text.split(/\s+/).filter(Boolean).map(name => name.split("@").join(suffix));
}

/** download-artifact pattern for exact file-named artifacts. @param {string[]} files */
function downloadPattern(files) {
  if (files.some(file => !/^[A-Za-z0-9._-]+$/.test(file))) {throw new Error("Native CI download names must be plain file names");}
  return files.length > 1 ? "{" + files.join(",") + "}" : files[0] || "";
}

/** Each upload is "file" (archive:false, artifact name == file name) or "name=path" (zip archive).
 * Relative paths are under RUNNER_TEMP. @param {string[]} rows @param {string} runnerTemp */
function uploadPlan(rows, runnerTemp) {
  if (rows.length > maxUploads) {throw new Error("Native CI jobs upload at most " + maxUploads + " artifacts");}
  return rows.map(row => {
    const [name = "", file = ""] = row.includes("=") ? row.split("=", 2) : ["", row];
    if (!file || /[\s,{}]/.test(row) || !name && /[*?]/.test(file) || name && !/^[A-Za-z0-9._-]+$/.test(name)) {
      throw new Error("Native CI upload is malformed: " + row);
    }
    return {name, path: path.isAbsolute(file) ? file : path.join(runnerTemp, file)};
  });
}

/** Write the job environment and transfer plan for the workflow. @param {NodeJS.ProcessEnv} environment */
function prepareJob(environment) {
  const envFile = environment["GITHUB_ENV"] || "", outputFile = environment["GITHUB_OUTPUT"] || "",
    runnerTemp = environment["RUNNER_TEMP"] || "", suffix = runSuffix(environment);
  if (![envFile, outputFile, runnerTemp].every(file => path.isAbsolute(file))) {throw new Error("Native CI job needs GITHUB_ENV, GITHUB_OUTPUT and RUNNER_TEMP");}
  const pins = parsePins(environment["NATIVE_JOB_PINS"] || ""), values = {...nativeCiEnvironment({...environment, ...pins}), ...pins},
    uploads = uploadPlan(names(environment["NATIVE_JOB_UPLOAD"] || "", suffix), runnerTemp),
    diagnostics = uploadPlan(names(environment["NATIVE_JOB_DIAGNOSTICS"] || "", suffix), runnerTemp);
  if (diagnostics.length > 1 || diagnostics.some(row => !row.name)) {throw new Error("Native CI diagnostics must be one name=path entry");}
  /** @type {Record<string, string>} */
  const outputs = {pattern: downloadPattern(names(environment["NATIVE_JOB_DOWNLOAD"] || "", suffix)),
    diagnostics_name: diagnostics[0]?.name || "", diagnostics_path: diagnostics[0]?.path || ""};
  uploads.forEach((row, index) => {
    outputs[`upload_${index + 1}`] = row.path;
    outputs[`upload_${index + 1}_name`] = row.name;
  });
  fs.appendFileSync(envFile, Object.entries(values).map(([key, value]) => key + "=" + value).join("\n") + "\n", "utf8");
  fs.appendFileSync(outputFile, Object.entries(outputs).map(([key, value]) => key + "=" + value).join("\n") + "\n", "utf8");
  return {values, outputs};
}

module.exports = {roots, runSuffix, nativeCiEnvironment, parsePins, names, downloadPattern, uploadPlan, prepareJob};

if (require.main === module) {
  try {
    if (process.argv.length !== 2) {throw new Error("Usage: job.js (inside a native workflow job)");}
    prepareJob(process.env);
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
