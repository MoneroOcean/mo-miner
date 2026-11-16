"use strict";

const path = require("node:path");
const fs = require("node:fs");
const {createRequire} = require("node:module");

/** Remove only the contiguous scaffold installed by npm run-script's set-path.js.
 * Consuming its package anchor makes subsequent native build boundaries idempotent;
 * a genuine nested npm invocation supplies a new anchor.
 * @param {NodeJS.ProcessEnv} env
 * @param {NodeJS.Platform} [platform]
 * @returns {NodeJS.ProcessEnv}
 */
function normalizeBuildEnvironment(env, platform = process.platform) {
  const result = {...env};
  const paths = Object.keys(env).filter(key => platform === "win32" ? /^path$/i.test(key) : key === "PATH");
  const firstPath = paths[0];
  const value = firstPath ? env[firstPath] : undefined;
  const packageJson = env["npm_package_json"];
  const npmCli = env["npm_execpath"];
  const api = platform === "win32" ? path.win32 : path.posix;
  if (!value || paths.some(key => env[key] !== value) || !packageJson || !npmCli ||
      !env["npm_lifecycle_event"] || !api.isAbsolute(packageJson) || !api.isAbsolute(npmCli) ||
      api.basename(packageJson) !== "package.json" || api.basename(npmCli) !== "npm-cli.js" ||
      [value, packageJson, npmCli].some(item => /[\r\n\0]/.test(item))) {
    return result;
  }

  /** @type {string[]} */
  const scaffold = [];
  let directory = api.dirname(packageJson);
  while (true) {
    scaffold.push(api.resolve(directory, "node_modules", ".bin"));
    const parent = api.dirname(directory);
    if (parent === directory) {
      break;
    }
    directory = parent;
  }
  let nodeGypBin = api.resolve(api.dirname(npmCli), "../node_modules/@npmcli/run-script/lib/node-gyp-bin");
  if (platform === process.platform) {
    try {
      // Distro npm can share dependencies outside its own node_modules. Resolve metadata
      // from the actual CLI without loading package code; synthetic CLI paths keep the fallback.
      if (fs.statSync(npmCli).isFile()) {
        const metadata = createRequire(npmCli).resolve("@npmcli/run-script/package.json");
        nodeGypBin = api.resolve(api.dirname(metadata), "lib/node-gyp-bin");
      }
    } catch { /* Unavailable npm metadata leaves the conventional layout unchanged. */ }
  }
  scaffold.push(nodeGypBin);
  const delimiter = platform === "win32" ? ";" : ":";
  const entries = value.split(delimiter);
  const equal = (/** @type {string | undefined} */ actual, /** @type {string} */ expected) =>
    actual !== undefined && (platform === "win32" ? actual.toLowerCase() === expected.toLowerCase() : actual === expected);
  const matches = (/** @type {number} */ offset) => scaffold.every((entry, index) => equal(entries[offset + index], entry));
  // A second identical scaffold could be deliberate caller PATH, or an unnormalized
  // outer npm invocation. Without that provenance neither layer can safely be removed.
  if (!matches(0) || matches(scaffold.length)) {
    return result;
  }
  const original = entries.slice(scaffold.length).join(delimiter);
  for (const key of paths) {
    result[key] = original;
  }
  delete result["npm_package_json"];
  return result;
}

/** One release policy for archive and static-native builds. SDK activation may select tools,
 * but must not silently reuse host-only objects or omit policy-selected workers.
 * @param {NodeJS.ProcessEnv} env @returns {NodeJS.ProcessEnv} */
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
  return {...normalizeBuildEnvironment(env), MOM_GPU_BACKEND: "all", MOM_LINUX_BUILD_COMPILER: "all",
    MOM_PORTABLE_BUILD: "1", MOM_REUSE_BUILT_WORKER: "0"};
}

/** Pinned SDK source tools must not inherit Git replacement/attribute hooks or Tar options.
 * Child-only sanitization leaves the developer's checkouts and account settings intact.
 * @param {NodeJS.ProcessEnv} environment */
function sdkSourceEnvironment(environment) {
  const env = releaseBuildEnvironment(environment);
  for (const key of Object.keys(env)) {if (key.startsWith("GIT_") || key === "TAR_OPTIONS") {delete env[key];}}
  return Object.assign(env, {GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1", GIT_OPTIONAL_LOCKS: "0", GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null", GIT_ATTR_NOSYSTEM: "1", LC_ALL: "C"});
}

/** Native SDK builds must not inherit compiler/setup hooks. Windows INCLUDE/LIB remain
 * scoped to the selected SDK, while all commands still use explicit compiler paths.
 * @param {NodeJS.ProcessEnv} environment */
function nativeSdkEnvironment(environment) {
  const env = sdkSourceEnvironment(environment);
  for (const key of Object.keys(env)) {
    if (/^(?:CMAKE_.*|CC|CXX|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|CL|_CL_|LINK|_LINK_|PYTHONPATH|PYTHONHOME|BASH_ENV|BASH_FUNC_.*|ENV|LD_PRELOAD|DYLD_INSERT_LIBRARIES)$/i.test(key)) {delete env[key];}
  }
  return env;
}

module.exports = {normalizeBuildEnvironment, releaseBuildEnvironment, sdkSourceEnvironment, nativeSdkEnvironment};

if (require.main === module) {
  const mode = process.argv[2];
  if (process.argv.length !== 3 || !["--path", "--json"].includes(mode || "")) {
    process.stderr.write("Expected --path or --json\n");
    process.exitCode = 1;
  } else {
    const normalized = normalizeBuildEnvironment(process.env);
    if (normalized["npm_package_json"] === process.env["npm_package_json"]) {
      process.exitCode = 3;
    } else {
      const key = Object.keys(normalized).find(name => process.platform === "win32" ? /^path$/i.test(name) : name === "PATH");
      const value = key ? normalized[key] || "" : "";
      // PS5.1 native decoding is code-page dependent: JSON stdout must stay ASCII.
      process.stdout.write(mode === "--json" ? JSON.stringify(value).replace(/[\u007f-\uffff]/g,
        char => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`) : value);
    }
  }
}
