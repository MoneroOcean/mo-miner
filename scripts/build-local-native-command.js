"use strict";

const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const local = require("./build-local-native-sdk"), {fileHash} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** Install only the exact archive/checksum pair after the shared package audit succeeds.
 * Callers hold the build lock. Private same-filesystem staging preserves existing outputs;
 * exclusive hard-link installation also refuses uncoordinated destination-file creation.
 * @param {{archive:{path:string,sha256:string},sidecar:string}} result @param {string} destination */
function publishLocalDistribution(result, destination) {
  if (!path.isAbsolute(destination) || fs.realpathSync(destination) !== destination) {throw new Error("Release destination must be canonical");}
  for (const file of [result.archive.path, result.sidecar]) {
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file || stat.nlink !== 1) {
      throw new Error("Release input must be a canonical regular file: " + file);
    }
  }
  const name = path.basename(result.archive.path), sidecarName = name + ".sha256";
  if (!/^mom-v[0-9][0-9A-Za-z.-]*-(?:lin\.tgz|win\.zip)$/.test(name) ||
      !/^[a-f0-9]{64}$/.test(result.archive.sha256) || fileHash(result.archive.path) !== result.archive.sha256 ||
      fs.readFileSync(result.sidecar, "utf8") !== `${result.archive.sha256}  ${name}\n`) {
    throw new Error("Local release publication requires the verified archive and exact checksum sidecar");
  }
  const previous = new Map();
  for (const item of [name, sidecarName]) {
    const stat = fs.lstatSync(path.join(destination, item), {throwIfNoEntry: false});
    if (stat && (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1)) {
      throw new Error("Refusing to replace a non-regular release output: " + item);
    }
    previous.set(item, stat);
  }
  const stage = fs.mkdtempSync(path.join(destination, "mom-release-publish-"));
  let complete = false;
  /** @type {string[]} */ const saved = [];
  /** @type {{name:string,stat:fs.Stats}[]} */ const installed = [];
  try {
    fs.copyFileSync(result.archive.path, path.join(stage, name), fs.constants.COPYFILE_EXCL);
    fs.copyFileSync(result.sidecar, path.join(stage, sidecarName), fs.constants.COPYFILE_EXCL);
    if (fileHash(path.join(stage, name)) !== result.archive.sha256 ||
        fs.readFileSync(path.join(stage, sidecarName), "utf8") !== `${result.archive.sha256}  ${name}\n`) {
      throw new Error("Release archive/checksum changed while copying");
    }
    // Withdraw the old checksum first; interruption must not leave it certifying new bytes.
    for (const item of [sidecarName, name]) {
      const target = path.join(destination, item);
      const current = fs.lstatSync(target, {throwIfNoEntry: false}), prior = previous.get(item);
      if (Boolean(current) !== Boolean(prior) || current && prior &&
          ["dev", "ino", "size", "mtimeMs", "ctimeMs", "nlink"].some(key => Reflect.get(current, key) !== Reflect.get(prior, key))) {
        throw new Error("Release output changed during publication: " + item);
      }
      if (current) {
        const backup = path.join(stage, "old-" + item);
        fs.renameSync(target, backup); saved.push(item);
        const moved = fs.lstatSync(backup);
        if (["dev", "ino", "size", "mtimeMs", "nlink"].some(key => Reflect.get(moved, key) !== Reflect.get(current, key))) {
          throw new Error("Release output changed while withdrawing: " + item);
        }
      }
    }
    for (const item of [name, sidecarName]) {
      const source = path.join(stage, item), stat = fs.lstatSync(source);
      fs.linkSync(source, path.join(destination, item)); installed.push({name: item, stat});
      fs.unlinkSync(source);
    }
    complete = true;
  } catch (error) {
    try {
      for (const item of installed.reverse()) {
        const target = path.join(destination, item.name), current = fs.lstatSync(target, {throwIfNoEntry: false});
        if (!current || current.dev !== item.stat.dev || current.ino !== item.stat.ino ||
            current.size !== item.stat.size || current.mtimeMs !== item.stat.mtimeMs) {
          throw new Error("Release output changed during recovery: " + item.name, {cause: error});
        }
        fs.unlinkSync(target);
      }
      for (const item of saved.reverse()) {
        const target = path.join(destination, item);
        if (fs.lstatSync(target, {throwIfNoEntry: false})) {throw new Error("Refusing to overwrite a foreign output during recovery: " + item, {cause: error});}
        const source = path.join(stage, "old-" + item);
        fs.linkSync(source, target); fs.unlinkSync(source);
      }
      complete = true;
    } catch (recovery) {throw new AggregateError([error, recovery], "Local release recovery failed; preserved originals at " + stage, {cause: recovery});}
    throw error;
  } finally {if (complete) {fs.rmSync(stage, {recursive: true, force: true});}}
  return {archive: path.join(destination, name), sidecar: path.join(destination, sidecarName), sha256: result.archive.sha256};
}

/** SDK preparation, application linking and packing share one verified host-side limit.
 * SDK stages retain their admission checks and incremental caches without escaping that scope.
 * @param {string} workspace */
function runLocalNativeRelease(workspace) {
  local.verifyCurrentMemoryScope();
  require("./build-release").releaseVersion(process.env);
  local.admitWorkspace(workspace, repo);
  const stage = path.join(workspace, "release-work"), env = {...process.env, MOM_LOCAL_NATIVE_SCOPE: "1"};
  // One slot bounds failed-link disk growth. Never erase an earlier run's diagnostics.
  if (fs.lstatSync(stage, {throwIfNoEntry: false})) {throw new Error("Inspect preserved release work before retrying: " + stage);}
  fs.mkdirSync(stage, {mode: 0o700});
  let published = false;
  try {
    // The SDK producers and the application link both use the pinned glibc sysroot.
    const sysrootEnv = local.ensureLocalLinuxSysroot(workspace, env),
      sdk = require("./build-local-native-linux-sdk").buildLocalLinuxSdk({workspace}, {environment: sysrootEnv}),
      result = require("./build-local-native-distribution").buildLocalNativeDistribution(sdk, path.join(stage, "candidate"), sysrootEnv),
      output = publishLocalDistribution(result, repo);
    published = true;
    console.log(`[native] Built ${output.archive}\n[native] SHA256 ${output.sha256}; build audited, not runtime/publication qualified.`);
    return output;
  } finally {
    // Only this invocation's fresh private directory is discarded, never shared object/SDK caches.
    if (published) {fs.rmSync(stage, {recursive: true});} else if (fs.readdirSync(stage).length === 0) {fs.rmdirSync(stage);} else {
      console.error("[native] Preserved failed release work: " + stage);
    }
  }
}

/** The existing Engineering lock also protects its Core cache when the native frontdoor uses it.
 * @param {NodeJS.ProcessEnv} [environment]
 * @param {(command:string,args:string[],options:import('node:child_process').SpawnSyncOptions)=>{status:number|null,error?:Error}} [spawn]
 * @param {{platform?:NodeJS.Platform,arch?:string,launchWindows?:(env:NodeJS.ProcessEnv,launch:typeof spawn)=>number}} [dependencies] */
function launchLocalNativeRelease(environment = process.env, spawn = spawnSync, dependencies = {}) {
  const env = nativeSdkEnvironment(environment);
  require("./build-release").releaseVersion(env);
  const platform = dependencies.platform || process.platform, arch = dependencies.arch || process.arch;
  if (platform === "win32" && arch === "x64") {
    return (dependencies.launchWindows || require("./build-local-native-windows-command").launchLocalNativeWindowsRelease)(env, spawn);
  }
  if (platform !== "linux" || arch !== "x64") {
    throw new Error("Automatic local release preparation requires Linux x64 or Windows x64 with WSL2");
  }
  // Fail before creating caches or acquiring sources when the job setting or installed prerequisites are invalid.
  const jobs = String(local.localBuildJobs(env)), memoryMax = local.localMemoryMax(env);
  require("./build-local-native-linux-sdk").preflightLocalLinuxSdk({environment: env});
  const tools = local.localTools(env), workspace = local.defaultWorkspace(),
    lifecycle = [
      // A dev parent may already hold this lock across the nested npm invocation.
      // eslint-disable-next-line no-template-curly-in-string -- Established Bash build-lock markers.
      'if [ "${MOM_R_SH:-0}" != 1 ] && [ "${MOM_NATIVE_BUILD_LOCK:-0}" != 1 ]; then',
      "  lock_dir=$(git rev-parse --path-format=absolute --git-path mom-locks)",
      '  mkdir -p -m 700 "$lock_dir"',
      // eslint-disable-next-line no-template-curly-in-string -- Bash UID in the established Engineering lock name.
      '  exec 9>"$lock_dir/mom-r-sh-${UID}.build.lock"',
      "  flock 9", "  export MOM_NATIVE_BUILD_LOCK=1", "fi", 'exec "$@"',
    ].join("\n"),
    result = spawn("bash", ["-eu", "-c", lifecycle, "--", tools.systemdRun, "--user", "--scope", "--quiet",
      "--property=MemoryMax=" + memoryMax, "--property=MemorySwapMax=0", "--", process.execPath, __filename, "--build", workspace],
    {cwd: repo, env: {...env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: jobs}, stdio: "inherit"});
  if (result.error) {throw result.error;}
  return result.status ?? 1;
}

if (require.main === module) {
  try {
    const [command, workspace, extra] = process.argv.slice(2);
    if (command !== "--build" || !workspace || extra) {throw new Error("Internal local release command; use npm run build:release");}
    runLocalNativeRelease(workspace);
  } catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {publishLocalDistribution, runLocalNativeRelease, launchLocalNativeRelease};
