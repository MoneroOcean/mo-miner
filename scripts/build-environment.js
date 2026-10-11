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

module.exports = {normalizeBuildEnvironment};

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
