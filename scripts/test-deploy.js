"use strict";

const {spawn, spawnSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {normalizeBuildEnvironment} = require("./build-environment");

const repo = path.resolve(__dirname, "..");
const vendors = ["intel", "nvidia", "amd"];
const powershell = ["-NoProfile", "-ExecutionPolicy", "Bypass"];
// Inventory only: the existing release gates prove runtime and discrete-GPU usability.
const discovery = "$ErrorActionPreference='Stop'; " +
  "$ids=@(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { " +
  "if ([string]$_.PNPDeviceID -match 'VEN_(8086|10DE|1002)') { $Matches[1].ToUpperInvariant() } " +
  "} | Sort-Object -Unique); ConvertTo-Json -Compress -InputObject $ids";

/** Keep the public Node boundary alive until Bash finishes VM export/teardown.
 * @param {NodeJS.ProcessEnv} env
 * @param {(command: string, args: string[], options: import("node:child_process").SpawnOptions) =>
 *   import("node:child_process").ChildProcess} [launch]
 * @returns {Promise<number>}
 */
function deployLinux(env, launch = spawn) {
  return new Promise((resolve, reject) => {
    const child = launch("bash", [path.join(repo, "scripts", "test-deploy.sh")],
      {cwd: repo, env, stdio: "inherit"});
    const forwarded = new Set();
    /** @param {NodeJS.Signals} signal */
    const forward = signal => {
      if (!forwarded.has(signal)) {
        forwarded.add(signal);
        child.kill(signal);
      }
    };
    const interrupt = () => forward("SIGINT");
    const terminate = () => forward("SIGTERM");
    const cleanup = () => {
      process.off("SIGINT", interrupt);
      process.off("SIGTERM", terminate);
    };
    // Keep the caller's controlling terminal. Bash's interrupt handler is idempotent if
    // terminal delivery already reached it, and owns all escalation/cleanup deadlines.
    process.on("SIGINT", interrupt);
    process.on("SIGTERM", terminate);
    child.once("error", error => {cleanup(); reject(error);});
    child.once("close", (status, signal) => {
      cleanup();
      resolve(status ?? (signal === "SIGINT" ? 130 : signal === "SIGTERM" ? 143 : 1));
    });
  });
}

/** @param {NodeJS.ProcessEnv} env */
function windowsDeployOptions(env) {
  const target = env["MOM_DEPLOY_TARGET"] || "all";
  const selected = target === "all" || target === "windows" ? vendors :
    vendors.filter(vendor => target === `windows-${vendor}`);
  if (!selected.length) {throw new Error(`Unsupported native Windows MOM_DEPLOY_TARGET: ${target}`);}
  for (const name of ["MOM_DEPLOY_SKIP_VECTORS", "MOM_DEPLOY_REUSE_ARCHIVE", "MOM_FORCE_REBUILD"]) {
    if (env[name] && !["0", "1"].includes(env[name])) {throw new Error(`${name} must be 0 or 1`);}
  }
  if (env["MOM_FORCE_REBUILD"] === "1" && env["MOM_DEPLOY_REUSE_ARCHIVE"] === "1") {
    throw new Error("MOM_FORCE_REBUILD cannot be combined with MOM_DEPLOY_REUSE_ARCHIVE");
  }
  const version = (env["MOM_RELEASE_VERSION"] || require("../package.json").version).replace(/^[vV]/, "");
  if (!/^[0-9][0-9A-Za-z.-]*$/.test(version)) {throw new Error(`Invalid release version: ${version}`);}
  return {selected, multi: target === "all" || target === "windows",
    archive: path.join(repo, `mom-v${version}-win.zip`)};
}

/** Each gate starts outside any inherited worker/runtime binding.
 * @param {NodeJS.ProcessEnv} env
 */
function gateEnvironment(env) {
  /** @type {NodeJS.ProcessEnv} */
  const clean = {...env, MOM_SKIP_MSR: "1"};
  for (const name of Object.keys(clean)) {
    if (/^(MOM_(NATIVE_|RUNTIME_|GPU_TEST_|REQUIRE_.*GPU_TESTS)|MOM_RELEASE_ARCHIVE_VALIDATION_ONLY$|MOM_GPU_BACKEND$|MOM_OPENCL_DEVICE_TYPE$|ONEAPI_DEVICE_SELECTOR$|ZE_AFFINITY_MASK$|ACPP_VISIBILITY_MASK$|UR_ADAPTERS_FORCE_LOAD$|OCL_ICD_FILENAMES$)/i.test(name)) {
      delete clean[name];
    }
  }
  return clean;
}

/**
 * @typedef {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptions) =>
 *   {status: number | null, error?: Error, stdout?: string | Buffer | null}} DeploySpawn
 */

/** @param {NodeJS.Platform} platform
 * @param {string} arch
 * @param {NodeJS.ProcessEnv} env
 * @param {DeploySpawn} [spawn]
 * @param {(file: string) => boolean} [exists]
 * @param {(line: string) => void} [report]
 */
function deploy(platform, arch, env, spawn = spawnSync, exists = fs.existsSync, report = console.log) {
  if (platform !== "win32" || arch !== "x64") {
    throw new Error("Native deployment supports Linux and Windows x86-64 only");
  }
  const options = windowsDeployOptions(env);
  const baseEnv = normalizeBuildEnvironment(env, platform);
  /** @param {string} command @param {string[]} args @param {NodeJS.ProcessEnv} childEnv */
  const run = (command, args, childEnv) => {
    const result = spawn(command, args, {cwd: repo, env: childEnv, stdio: "inherit"});
    if (result.error) {report(result.error.message);}
    return result.error ? 1 : result.status ?? 1;
  };
  if (!(env["MOM_DEPLOY_REUSE_ARCHIVE"] === "1" && exists(options.archive))) {
    const status = run(process.execPath, [path.join(repo, "scripts", "build-release.js")], baseEnv);
    if (status !== 0) {report("FAIL windows release build; GPU lanes not run"); return status;}
  } else {
    report(`Reusing ${path.basename(options.archive)}`);
  }
  if (!exists(options.archive)) {throw new Error(`Windows release archive missing: ${options.archive}`);}
  const inventory = spawn("powershell.exe", [...powershell, "-Command", discovery],
    {cwd: repo, env: baseEnv, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"]});
  if (inventory.error) {throw inventory.error;}
  if (inventory.status !== 0) {throw new Error("Windows GPU inventory failed");}
  const ids = JSON.parse(String(inventory.stdout || "").replace(/^\uFEFF/, ""));
  if (!Array.isArray(ids) || ids.some(id => !["8086", "10DE", "1002"].includes(id))) {
    throw new Error("Invalid Windows GPU inventory");
  }
  const present = vendors.filter((_vendor, index) => ids.includes(["8086", "10DE", "1002"][index]));
  let failed = 0;
  for (const vendor of options.selected) {
    if (!present.includes(vendor)) {report(`SKIP ${vendor}-windows: no PCI GPU detected`); continue;}
    const status = run("powershell.exe", [...powershell, "-File",
      path.join(repo, "scripts", "test-windows-release-gpu.ps1"),
      "-Archive", path.relative(repo, options.archive), "-SkipAbsentGpu", "-Platform", `${vendor}-windows`], gateEnvironment(baseEnv));
    // Only the helper's validated native absence report uses exit 3; never count it as PASS.
    if (status === 3) {
      present.splice(present.indexOf(vendor), 1);
      report(`SKIP ${vendor}-windows: no usable discrete GPU discovered`);
      continue;
    }
    report(`${status === 0 ? "PASS" : "FAIL"} ${vendor}-windows`);
    if (status !== 0) {failed = failed || status;}
  }
  if (options.multi) {
    const reason = env["MOM_DEPLOY_SKIP_VECTORS"] === "1" ? "vector gates disabled" :
      env["MOM_DEPLOY_ALGO"] ? "single-algorithm deployment" :
        present.length < 2 ? "fewer than two discrete GPU vendors available" : "";
    if (reason) {report(`SKIP multi-windows: ${reason}`);} else {
      const status = run("powershell.exe", [...powershell, "-File",
        path.join(repo, ".github", "workflows", "scripts", "test-release-windows.ps1"),
        "-Archive", path.relative(repo, options.archive), "-Suite", "gpu-multi"],
      {...gateEnvironment(baseEnv), MOM_GPU_BACKEND: present[0],
        MOM_GPU_TEST_VENDORS: present.join(","), MOM_REQUIRE_MULTI_GPU_TESTS: "1"});
      report(`${status === 0 ? "PASS" : "FAIL"} multi-windows`);
      if (status !== 0) {failed = failed || status;}
    }
  }
  return failed;
}

if (require.main === module) {
  (async () => {
    if (process.argv.length !== 2) {throw new Error("Usage: npm run test:deploy");}
    process.exitCode = process.platform === "linux" ? await deployLinux(process.env) :
      deploy(process.platform, process.arch, process.env);
  })().catch(error => {
    console.error(error instanceof Error ? error.message : "Native deployment failed");
    process.exitCode = 1;
  });
}

module.exports = {deploy, deployLinux, windowsDeployOptions};
