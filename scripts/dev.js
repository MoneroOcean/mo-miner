"use strict";

const {spawnSync} = require("node:child_process");
const path = require("node:path");
const {normalizeBuildEnvironment} = require("./build-environment");
const {windowsCmdArgs} = require("./windows-command");

const repo = path.resolve(__dirname, "..");

/** @param {NodeJS.ProcessEnv} env @returns {NodeJS.ProcessEnv} */
function nativeDevEnvironment(env) {
  const value = env["MOM_GPU_BACKEND"] || "intel";
  const backend = /^(intel|intel_igpu)$/i.test(value) ? value.toLowerCase() : value;
  if (!["intel", "intel_igpu", "nvidia", "amd", "opencl", "all"].includes(backend)) {
    throw new Error(`Unknown MOM_GPU_BACKEND: ${backend}`);
  }
  const reuse = env["MOM_REUSE_BUILT_WORKER"] || "0";
  if (!["0", "1"].includes(reuse)) {
    throw new Error("MOM_REUSE_BUILT_WORKER must be 0 or 1");
  }
  if (reuse === "1") {
    // Container workers may have a different Node ABI/SDK; source-only reuse cannot validate that.
    throw new Error("MOM_REUSE_BUILT_WORKER=1 is unsupported in native development; unset it and use ordinary npm run dev, which validates incremental caches");
  }
  /** @type {NodeJS.ProcessEnv} */
  const result = {...env, MOM_GPU_BACKEND: backend, MOM_REUSE_BUILT_WORKER: reuse};
  if (!env["MOM_GPU_TEST_VENDORS"] && env["MOM_GPU_BACKEND"] !== undefined && backend !== "opencl") {
    result["MOM_GPU_TEST_VENDORS"] = backend === "all" ? "intel,nvidia,amd" :
      backend === "intel_igpu" ? "intel" : backend;
  }
  return result;
}

/** Resolve Node/npm without passing user arguments through cmd.exe.
 * @param {string[]} args
 * @param {NodeJS.ProcessEnv} env
 * @returns {string[]}
 */
function windowsDevCommand(args, env) {
  const [command, ...rest] = args;
  if (!command) {return [process.execPath, "mom.js"];}
  if (/^node(?:\.exe)?$/i.test(command)) {return [process.execPath, ...rest];}
  if (/^npm(?:\.cmd)?$/i.test(command)) {
    const npmCli = env["npm_execpath"];
    if (!npmCli || !path.win32.isAbsolute(npmCli) || path.win32.basename(npmCli) !== "npm-cli.js") {
      throw new Error("npm_execpath must identify npm-cli.js; invoke this task through npm run dev");
    }
    return [process.execPath, npmCli, ...rest];
  }
  if (/\.(?:cmd|bat)$/i.test(command)) {return ["cmd.exe", ...windowsCmdArgs(args)];}
  return args;
}

/** Let MoM's compiler policy select the control/algorithm addon and its matching runtime.
 * In particular, NVIDIA capability detection must happen before choosing an SM80 worker.
 * @param {NodeJS.ProcessEnv} env
 * @returns {NodeJS.ProcessEnv}
 */
function windowsRuntimeEnvironment(env) {
  return {...env, MOM_NATIVE_DIR: env["MOM_NATIVE_DIR"] || path.join(repo, "build", "win", "compilers")};
}

/**
 * @typedef {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptions) =>
 *   {status: number | null, error?: Error}} DevSpawn
 */

/** Linux defaults to Docker; MOM_DEV_HOST=1 uses the same builder with installed host SDKs.
 * Windows always shares the native incremental build/win tree.
 * @param {NodeJS.Platform} platform
 * @param {string} arch
 * @param {NodeJS.ProcessEnv} env
 * @param {string[]} args
 * @param {DevSpawn} [spawn]
 * @returns {number}
 */
function runDev(platform, arch, env, args, spawn = spawnSync) {
  const hostMode = env["MOM_DEV_HOST"] || "0";
  if (!["0", "1"].includes(hostMode)) {throw new Error("MOM_DEV_HOST must be 0 or 1");}
  const host = hostMode === "1";
  // Default to MoM arguments. Keep explicit Node/npm workflows; other tools use --exec.
  if (args[0] === "--exec") {
    args = args.slice(1);
    if (!args[0]) {throw new Error("--exec requires a command");}
  } else if (!/^(node(?:\.exe)?|npm(?:\.cmd)?)$/i.test(args[0] || "")) {
    args = ["node", "mom.js", ...args];
  }
  /** @type {string[][]} */
  const commands = [];
  let runtimeEnv = env;
  if (platform === "linux") {
    if (host) {
      if (arch !== "x64") {throw new Error("Native Linux development supports x86-64 only");}
      runtimeEnv = {...nativeDevEnvironment(env), MOM_DEV_HOST: "1"};
    }
    commands.push(["bash", path.join(repo, "scripts", host ? "multicompiler-entrypoint.sh" : "dev-container.sh"),
      ...args]);
  } else if (platform === "win32" && arch === "x64") {
    runtimeEnv = nativeDevEnvironment(env);
    // Resolve/validate the requested command before a potentially expensive build.
    const command = windowsDevCommand(args, runtimeEnv);
    const backend = runtimeEnv["MOM_GPU_BACKEND"] || "intel";
    commands.push(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
      path.join(repo, ".github", "workflows", "scripts", "build-windows-multicompiler.ps1"),
      "-Backend", backend]);
    commands.push(command);
  } else {
    throw new Error("Development supports Linux and native Windows x86-64 only");
  }
  for (const [index, invocation] of commands.entries()) {
    const [command, ...commandArgs] = invocation;
    if (!command) {throw new Error("Development command is missing");}
    const build = platform === "linux" && host || platform === "win32" && index === 0;
    const result = spawn(command, commandArgs, {cwd: repo,
      env: build ? normalizeBuildEnvironment(runtimeEnv, platform) :
        platform === "win32" ? windowsRuntimeEnvironment(runtimeEnv) : runtimeEnv, stdio: "inherit",
      ...(platform === "win32" && command === "cmd.exe" ? {windowsVerbatimArguments: true} : {})});
    if (result.error) {
      if (build && platform === "win32") {
        throw new Error(`Cannot start the Windows builder: ${result.error.message}. Install prerequisites with scripts\\install-dev.bat`);
      }
      throw result.error;
    }
    if (result.status !== 0) {return result.status ?? 1;}
  }
  return 0;
}

if (require.main === module) {
  try {
    process.exitCode = runDev(process.platform, process.arch, process.env, process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Development task failed");
    process.exitCode = 1;
  }
}

module.exports = {runDev, windowsDevCommand, nativeDevEnvironment, windowsRuntimeEnvironment};
