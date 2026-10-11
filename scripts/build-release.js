"use strict";

const {spawnSync} = require("node:child_process");
const path = require("node:path");
const {normalizeBuildEnvironment} = require("./build-environment");

const repo = path.resolve(__dirname, "..");
const version = require("../package.json").version;

/**
 * @param {NodeJS.Platform} platform
 * @param {boolean} [packageOnly]
 * @param {string} [releaseVersion]
 */
function releaseBuildCommands(platform, packageOnly = false, releaseVersion = version) {
  const scripts = path.join(repo, ".github", "workflows", "scripts");
  if (platform === "linux") {
    // Keep the existing cache lock through publication, including package-only runs.
    // Development parents already own this host lock around their entire command.
    const lifecycle = [
      // eslint-disable-next-line no-template-curly-in-string -- Bash parameter expansion.
      'if [ "${MOM_R_SH:-0}" != 1 ] && [ "${MOM_NATIVE_BUILD_LOCK:-0}" != 1 ]; then',
      "  lock_dir=$(git rev-parse --path-format=absolute --git-path mom-locks)",
      '  mkdir -p -m 700 "$lock_dir"',
      // eslint-disable-next-line no-template-curly-in-string -- Bash's numeric caller UID.
      '  exec 9>"$lock_dir/mom-r-sh-${UID}.build.lock"',
      "  flock 9",
      "  export MOM_NATIVE_BUILD_LOCK=1",
      "fi",
      'if [ "$4" != 1 ]; then bash "$1" true; fi',
      'exec bash "$2" "$3"',
    ].join("\n");
    return [["bash", "-eu", "-c", lifecycle, "--",
      path.join(repo, "scripts", "multicompiler-entrypoint.sh"),
      path.join(scripts, "package-linux-combined.sh"), releaseVersion, packageOnly ? "1" : "0"]];
  }
  if (platform === "win32") {
    const powershell = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"];
    const commands = [
      [...powershell, path.join(scripts, "build-windows-multicompiler.ps1"), "-Backend", "all"],
      [...powershell, path.join(scripts, "package-windows.ps1"), "-Version", releaseVersion],
    ];
    return packageOnly ? commands.slice(1) : commands;
  }
  throw new Error("Native release builds support Linux and Windows x86-64 only");
}

/** Normalize the deploy override before it can reach a command or archive path.
 * @param {NodeJS.ProcessEnv} env
 */
function releaseVersion(env) {
  const value = (env["MOM_RELEASE_VERSION"] || version).replace(/^[vV]/, "");
  if (!/^[0-9]/.test(value) || /[^0-9A-Za-z.-]/.test(value)) {
    throw new Error(`Invalid release version: ${value}`);
  }
  return value;
}

/**
 * @param {NodeJS.ProcessEnv} env
 * @returns {NodeJS.ProcessEnv}
 */
function releaseBuildEnvironment(env) {
  if (env["MOM_FORCE_REBUILD"] && !["0", "1"].includes(env["MOM_FORCE_REBUILD"])) {
    throw new Error("MOM_FORCE_REBUILD must be 0 or 1");
  }
  if (env["MOM_CPU_MARCH"]) {
    throw new Error("Unset MOM_CPU_MARCH before building a portable release");
  }
  if (env["MOM_LINUX_BUILD_PLAN_ONLY"] === "1") {
    throw new Error("Unset MOM_LINUX_BUILD_PLAN_ONLY before building a release");
  }
  for (const name of ["MOM_DPCPP_IMPL", "MOM_COMBINED_TARGETS"]) {
    if (env[name]) {
      throw new Error(`Unset ${name} before building the policy-selected release workers`);
    }
  }
  // A release always contains all portable workers, even in a targeted developer shell.
  return {...normalizeBuildEnvironment(env), MOM_GPU_BACKEND: "all", MOM_LINUX_BUILD_COMPILER: "all",
    MOM_PORTABLE_BUILD: "1", MOM_REUSE_BUILT_WORKER: "0"};
}

/**
 * @typedef {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptions) =>
 *   {status: number | null, error?: Error}} ReleaseSpawn
 */

/**
 * @param {NodeJS.Platform} platform
 * @param {string} arch
 * @param {NodeJS.ProcessEnv} env
 * @param {boolean} [packageOnly]
 * @param {ReleaseSpawn} [spawn]
 * @returns {number}
 */
function buildRelease(platform, arch, env, packageOnly = false, spawn = spawnSync) {
  if (arch !== "x64") {
    throw new Error("Native release builds support Linux and Windows x86-64 only");
  }
  const commands = releaseBuildCommands(platform, packageOnly, releaseVersion(env));
  const buildEnv = releaseBuildEnvironment(env);
  for (const [command, ...args] of commands) {
    if (!command) {throw new Error("Release build command is missing");}
    const result = spawn(command, args, {cwd: repo, env: buildEnv, stdio: "inherit"});
    if (result.error) {throw result.error;}
    if (result.status !== 0) {return result.status ?? 1;}
  }
  return 0;
}

if (require.main === module) {
  try {
    const args = process.argv.slice(2);
    if (args.length > 1 || args.length === 1 && !["--package-only", "--clean"].includes(args[0] || "")) {
      throw new Error("Usage: npm run build:release [-- --package-only | --clean]");
    }
    const env = args[0] === "--clean" ? {...process.env, MOM_FORCE_REBUILD: "1"} : process.env;
    process.exitCode = buildRelease(process.platform, process.arch, env, args[0] === "--package-only");
    if (process.exitCode === 0) {
      console.log(`Built mom-v${releaseVersion(process.env)}-${process.platform === "win32" ? "win.zip" : "lin.tgz"}`);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Native release build failed");
    process.exitCode = 1;
  }
}

module.exports = {buildRelease, releaseBuildCommands, releaseBuildEnvironment};
