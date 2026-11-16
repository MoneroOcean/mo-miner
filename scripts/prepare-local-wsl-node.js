"use strict";

// Keep Windows node_modules usable by Engineering Mode. WSL gets a separate,
// lockfile-pinned dependency generation; only its Linux esbuild executable is selected.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), {dependencyPins} = require("./check-native-notices"),
  {fileHash, readBuildJson} = require("./build-source-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** @param {string} file */
function canonicalFile(file) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file) {
    throw new Error("WSL build dependency is not a canonical regular file: " + file);
  }
  return stat;
}

/** Prepare once, then rehash without npm/network or changes to the Windows checkout.
 * The outer coordinator owns the build lock and the verified aggregate memory limit.
 * @param {string} workspace
 * @param {{verifyScope?:()=>unknown,nodeVersion?:string,npmCli?:string,run?:typeof spawnSync,environment?:NodeJS.ProcessEnv}} [dependencies] */
function prepareLocalWslNode(workspace, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("WSL build dependencies require Linux x64");}
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  const nodeVersion = dependencies.nodeVersion || process.version;
  if (nodeVersion !== dependencyPins().buildNode.version) {throw new Error("WSL requires pinned build Node.js " + dependencyPins().buildNode.version);}
  local.admitWorkspace(workspace, repo);
  const packageFile = path.join(repo, "package.json"), lockFile = path.join(repo, "package-lock.json"),
    packageBytes = fs.readFileSync(packageFile), lockBytes = fs.readFileSync(lockFile), pkg = JSON.parse(packageBytes.toString("utf8")),
    lock = JSON.parse(lockBytes.toString("utf8")), npmCli = fs.realpathSync(dependencies.npmCli || path.join(path.dirname(process.execPath), "npm")),
    version = pkg.devDependencies?.esbuild, platformPackage = lock.packages?.["node_modules/@esbuild/linux-x64"];
  canonicalFile(npmCli);
  if (!/^\d+\.\d+\.\d+$/.test(version || "") || lock.packages?.["node_modules/esbuild"]?.version !== version ||
      platformPackage?.version !== version || !/^sha512-[A-Za-z0-9+/]+=*$/.test(platformPackage.integrity || "")) {
    throw new Error("WSL esbuild dependency must have its exact version and platform integrity in the maintained lockfile");
  }
  const inputs = {package: fileHash(packageFile), lock: fileHash(lockFile), npm: {path: npmCli, sha256: fileHash(npmCli)},
      node: nodeVersion, recipe: fileHash(__filename)},
    key = createHash("sha256").update(JSON.stringify(inputs)).digest("hex"), parent = path.join(workspace, "wsl-node-dependencies"),
    root = path.join(parent, key), binary = path.join(root, "node_modules/@esbuild/linux-x64/bin/esbuild"),
    receipt = path.join(root, "admitted.json"), existing = fs.lstatSync(root, {throwIfNoEntry: false}),
    run = dependencies.run || spawnSync;
  if (!fs.existsSync(parent)) {fs.mkdirSync(parent, {mode: 0o700});}
  if (!fs.lstatSync(parent).isDirectory() || fs.lstatSync(parent).isSymbolicLink() || fs.realpathSync(parent) !== parent) {
    throw new Error("WSL dependency cache parent is redirected");
  }
  if (existing && (!existing.isDirectory() || existing.isSymbolicLink() || fs.realpathSync(root) !== root)) {
    throw new Error("WSL dependency generation is redirected");
  }
  if (!existing) {
    fs.mkdirSync(root, {mode: 0o700});
    fs.writeFileSync(path.join(root, "package.json"), packageBytes, {flag: "wx"});
    fs.writeFileSync(path.join(root, "package-lock.json"), lockBytes, {flag: "wx"});
    // npm refuses one path loaded twice as user/global configuration.
    const userConfig = path.join(root, "npm-userconfig"), globalConfig = path.join(root, "npm-globalconfig");
    for (const file of [userConfig, globalConfig]) {fs.writeFileSync(file, "", {flag: "wx", mode: 0o600});}
    const cache = path.join(root, "npm-cache"), env = {...process.env, ...(dependencies.environment || {}),
      PATH: [path.dirname(process.execPath), "/usr/bin", "/bin"].join(":"), npm_config_logs_max: "1"};
    const result = run(process.execPath, [npmCli, "ci", "--prefix", root, "--ignore-scripts", "--include=dev", "--include=optional",
      "--no-audit", "--no-fund", "--userconfig=" + userConfig, "--globalconfig=" + globalConfig, "--cache", cache],
    {cwd: root, env, stdio: "inherit", timeout: 10 * 60 * 1000});
    if (result.error || result.signal || result.status !== 0) {throw new Error("WSL dependency installation failed; preserve " + root, {cause: result.error});}
    for (const file of [userConfig, globalConfig]) {
      if (canonicalFile(file).size !== 0) {throw new Error("Private npm config changed during dependency acquisition");}
      fs.unlinkSync(file);
    }
    // Only this fresh invocation's reproducible download cache is removed, after npm exits.
    if (fs.existsSync(cache)) {
      const cacheStat = fs.lstatSync(cache);
      if (!cacheStat.isDirectory() || cacheStat.isSymbolicLink() || fs.realpathSync(cache) !== cache) {
        throw new Error("WSL npm download cache is redirected; preserve it");
      }
      fs.rmSync(cache, {recursive: true});
    }
    const stat = canonicalFile(binary), probe = run(binary, ["--version"], {encoding: "utf8", timeout: 10000, maxBuffer: 4096});
    if (!(stat.mode & 0o111) || stat.size > 32 << 20 || probe.error || probe.signal || probe.status !== 0 || String(probe.stdout).trim() !== version) {
      throw new Error("WSL bundler is not the pinned Linux esbuild executable");
    }
    fs.writeFileSync(receipt, JSON.stringify({version: 1, inputs, binary: {path: binary, sha256: fileHash(binary), bytes: stat.size}}, null, 2) + "\n", {flag: "wx", mode: 0o444});
  }
  if (!isDeepStrictEqual(fs.readdirSync(root).sort(), ["admitted.json", "node_modules", "package-lock.json", "package.json"])) {
    throw new Error("WSL dependency generation is incomplete or contains unknown state; preserve it");
  }
  canonicalFile(receipt);
  const record = readBuildJson(receipt), stat = canonicalFile(binary);
  for (const [name, expected] of /** @type {Array<[string,string]>} */ ([["package.json", inputs.package], ["package-lock.json", inputs.lock]])) {
    canonicalFile(path.join(root, name));
    if (fileHash(path.join(root, name)) !== expected) {throw new Error("WSL dependency source copy changed: " + name);}
  }
  if (record.version !== 1 || !isDeepStrictEqual(record.inputs, inputs) || record.binary?.path !== binary ||
      record.binary.sha256 !== fileHash(binary) || record.binary.bytes !== stat.size || !(stat.mode & 0o111) ||
      fileHash(packageFile) !== inputs.package || fileHash(lockFile) !== inputs.lock || fileHash(npmCli) !== inputs.npm.sha256) {
    throw new Error("WSL build dependency or original lock/tool inputs changed");
  }
  return {root, binary, sha256: record.binary.sha256, version, reused: Boolean(existing)};
}

module.exports = {prepareLocalWslNode};
