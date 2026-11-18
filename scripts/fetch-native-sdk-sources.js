"use strict";

// Fetch pinned SDK inputs; compilation and extraction stay separate. Vendor binaries are
// explicitly named cohorts, never presented as source-built components.
const fs = require("node:fs"), path = require("node:path"), {spawn, spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {Readable, Transform} = require("node:stream"), {pipeline} = require("node:stream/promises");
const {setTimeout: sleep} = require("node:timers/promises");
const {sdkComponents, sdkSourceEnvironment} = require("./prepare-native-sdk-sources");
const {fileHash, sourceInputs, readBuildText} = require("./build-source-inputs");
const {ioPins} = require("./build-native-io");
const {archiveName: llvmArchiveName, archiveSha256: llvmArchiveSha256} = require("./native-llvm-sdk");
const {opensslPin, opensslBuildRecipe} = require("./build-native-openssl");
const {hermesPins} = require("./build-native-hermes");
const {zstdPin, zstdBuildRecipe} = require("./build-native-zstd");
const {linuxSysrootPackages, linuxSysrootRelease, linuxSysrootOrigins} = require("./native-linux-sysroot");
const {libdrmPin, libdrmBuildRecipe} = require("./build-native-libdrm");

const repo = path.resolve(__dirname, ".."), origins = Object.freeze({
  dpcpp: "https://github.com/intel/llvm.git", emhash: "https://github.com/ktprime/emhash.git",
  umf: "https://github.com/oneapi-src/unified-memory-framework.git", levelZero: "https://github.com/oneapi-src/level-zero.git",
  computeHeaders: "https://github.com/intel/compute-runtime.git", hwloc: "https://github.com/open-mpi/hwloc.git",
  openclLoader: "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git", openclHeaders: "https://github.com/KhronosGroup/OpenCL-Headers.git",
  vcIntrinsics: "https://github.com/intel/vc-intrinsics.git", spirvHeaders: "https://github.com/KhronosGroup/SPIRV-Headers.git",
  rocmLlvm: "https://github.com/ROCm/llvm-project.git", clr: "https://github.com/ROCm/clr.git",
  hip: "https://github.com/ROCm/hip.git", rocr: "https://github.com/ROCm/ROCR-Runtime.git",
  hermes: "https://github.com/facebook/hermes.git", acpp: "https://github.com/AdaptiveCpp/AdaptiveCpp.git",
});
const ioArchiveUrls = Object.freeze({
  libuv: "https://dist.libuv.org/dist/v1.51.0/libuv-v1.51.0.tar.gz",
  // Use upstream's byte-identical release asset as the stable CI download endpoint.
  zlib: "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz",
});
const ioArchiveOrigins = Object.freeze({libuv: ["https://dist.libuv.org"],
  zlib: ["https://github.com", "https://release-assets.githubusercontent.com"]});
const maxIoArchiveBytes = 128 << 20, ioArchiveTimeoutMs = 10 * 60 * 1000, maxArchiveRedirects = 3;
const buildArchiveUrls = Object.freeze({
  llvm: "https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/llvm-project-21.1.8.src.tar.xz",
  openssl: "https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz",
});
const buildArchiveOrigins = Object.freeze(["https://github.com", "https://release-assets.githubusercontent.com"]);
const maxBuildArchiveBytes = 192 << 20, buildArchiveTimeoutMs = 10 * 60 * 1000;
const maxWindowsOpenSslArchiveBytes = 128 << 20;
const zstdArchiveUrl = "https://github.com/facebook/zstd/releases/download/v" + zstdPin.version + "/zstd-" + zstdPin.version + ".tar.gz";
const maxZstdArchiveBytes = 16 << 20;
const maxSysrootArchiveBytes = 8 << 20;
const nvrtcArchiveOrigin = "https://developer.download.nvidia.com", maxNvrtcArchiveBytes = 320 << 20;
const windowsHostOrigin = "https://download.visualstudio.microsoft.com", maxWindowsHostArchiveBytes = 64 << 20;
// A transient network error, 429 or 5xx is retried after these delays; pinned hashes are never retried into success.
const archiveRetryDelaysMs = Object.freeze([2000, 8000]), gitFetchAttempts = 3;
/** @type {WeakSet<object>} */ const transientErrors = new WeakSet();

/** Mark an error as a transient network failure. @param {unknown} error */
function transient(error) {
  const marked = error instanceof Error ? error : new Error(String(error));
  transientErrors.add(marked);
  return marked;
}
const windowsHostLock = /** @type {WindowsHostSdkLock} */ (require("../runtime/windows-host-sdk.json"));
/** @typedef {{libuv: {directory: string, sha256: string}, zlib: {directory: string, sha256: string}}} IoArchivePins */
/** @typedef {Pick<Response, "status" | "url" | "body"> & {headers: Pick<Headers, "get">}} IoFetchResponse */
/** @typedef {(url: string, options: {redirect: "manual", signal: AbortSignal}) => Promise<IoFetchResponse>} IoArchiveFetch */
/** @typedef {Record<string, {sha256: string}>} ArchiveHashes */
/** @typedef {Record<string, {fileName: string, url: string, origins: readonly string[]}>} ArchiveSpecs */
/** @typedef {ReturnType<typeof ioSourceRecipe>} ArchiveRecipe */
/** @typedef {{target: string, sourceLabel: string, archiveLabel: string, specs: ArchiveSpecs, pins: ArchiveHashes,
 * maxBytes: number, timeoutMs: number, recipeFor: () => ArchiveRecipe, pinSnapshot: () => string, fetchImpl: IoArchiveFetch,
 * retryDelaysMs?: readonly number[]}} PinnedArchiveCohort */
/** @typedef {"llvm" | "openssl"} BuildArchiveName */
/** @typedef {"linux" | "windows" | "compiler" | "hermes" | "acpp" | "rocm"} GitSourceTarget */
/** @typedef {{version: number, target: string, status: string, networked: boolean, buildOnly: boolean, compiled: boolean,
 * publicationQualified: boolean, fetcherSha256: string, recipe: {sha256: string, files: Record<string, string>},
 * archives: Record<string, {path: string, url: string, sha256: string, bytes: number}>}} BuildArchiveReceipt */
/** Hermes/ACPP/ROCm remain separate Git cohorts because their producer inputs differ from DPC++.
 * Load each admission module only for its explicit source-fetch target.
 * @param {GitSourceTarget} target
 * @returns {Record<string, {revision: string, patches: string[]}>} */
function gitSourceComponents(target) {
  if (target === "hermes") {
    return {hermes: {revision: hermesPins.revision, patches: ["hermes-host-rejections.patch", "hermes-clang-cl.patch"]}};
  }
  if (target === "acpp") {
    const {syclRevision} = require("./native-sycl-inputs");
    return {acpp: {revision: syclRevision, patches: ["adaptivecpp-static.patch", "adaptivecpp-jit.patch"]}};
  }
  if (target === "rocm") {
    const {pins} = require("./build-native-rocm-host-runtime"), {revision, patches} = require("./native-comgr-inputs"),
      resourceSource = readBuildText(path.join(__dirname, "build-native-rocm-resources.js"), 1 << 20),
      clrRevision = [...resourceSource.matchAll(/^const clrRevision = "([0-9a-f]{40})";$/gm)][0]?.[1];
    // The HIPRTC and host-runtime producers pin CLR independently; reject source-cohort drift.
    if (!clrRevision || pins.clr !== clrRevision) {throw new Error("ROCm source consumers disagree on the pinned CLR revision");}
    return {rocmLlvm: {revision, patches: patches.map(file => path.basename(file))}, clr: {revision: pins.clr, patches: []},
      hip: {revision: pins.hip, patches: []}, rocr: {revision: pins.rocr, patches: []}};
  }
  return sdkComponents(target);
}

/** @param {GitSourceTarget} target */
function sourceRecipe(target) {
  const components = gitSourceComponents(target);
  let producers;
  if (target === "hermes") {producers = ["scripts/build-native-hermes.js"];} else if (target === "acpp") {
    producers = ["scripts/build-native-sycl-sdk.js", "scripts/prepare-native-sdk-sources.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-jit-link.js",
      "scripts/native-audit.js", "scripts/native-resources.js"];
  } else if (target === "rocm") {
    producers = ["scripts/build-native-comgr-sdk.js", "scripts/build-native-rocm-host-runtime.js", "scripts/build-native-rocm-resources.js",
      "scripts/native-comgr-inputs.js", "scripts/build-native-sycl-sdk.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-audit.js"];
  } else {producers = ["scripts/prepare-native-sdk-sources.js", "scripts/native-dpcpp-pins.js"];}
  const files = [...producers, "scripts/build-source-inputs.js", "scripts/build-environment.js",
    ...new Set(Object.values(components).flatMap(component => component.patches.map(name => "runtime/" + name)))];
  return sourceInputs(repo, [], files);
}

/** @param {string} file */
function executableGit(file) {
  try {if (!path.isAbsolute(file) || !fs.statSync(file).isFile()) {return false;} fs.accessSync(file, fs.constants.X_OK); return true;} catch {return false;}
}

/** @param {string} git @param {string[]} args @param {NodeJS.ProcessEnv} env @param {string} log @param {number} deadline @returns {Promise<void>} */
function runGit(git, args, env, log, deadline) {
  return new Promise((resolve, reject) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) {reject(new Error("10 minute SDK source component deadline exceeded")); return;}
    const fd = fs.openSync(log, "a"), child = spawn(git, args, {env, detached: true, stdio: ["ignore", fd, fd]});
    let timedOut = false, settled = false, killed = false;
    /** @type {number | null | undefined} */ let code;
    /** @type {NodeJS.Signals | null | undefined} */ let signal;
    /** @type {Error | undefined} */ let failure;
    /** @type {NodeJS.Timeout | undefined} */ let killTimer;
    const settle = () => {
      if (settled || (!failure && code === undefined) || (timedOut && !killed)) {return;}
      settled = true; clearTimeout(timer); if (!timedOut) {clearTimeout(killTimer);} fs.closeSync(fd);
      if (failure) {reject(failure); return;}
      if (timedOut || code !== 0) {reject(new Error(timedOut ? "10 minute SDK source component deadline exceeded" : `git failed (${code ?? signal}); log: ${log}`)); return;}
      resolve();
    };
    const timer = setTimeout(() => {
      timedOut = true;
      try {if (child.pid) {process.kill(-child.pid, "SIGTERM");} else {child.kill("SIGTERM");}} catch {child.kill("SIGTERM");}
      // Keep escalation alive after the Git leader exits; a descendant may ignore TERM.
      killTimer = setTimeout(() => {
        try {
          // detached gives this command a private PGID equal to its PID; signal only while that group exists.
          if (child.pid) {process.kill(-child.pid, 0); process.kill(-child.pid, "SIGKILL");} else {child.kill("SIGKILL");}
        } catch (error) {if (Reflect.get(Object(error), "code") !== "ESRCH") {failure = error instanceof Error ? error : new Error("Cannot kill timed-out Git process group");}}
        killed = true; settle();
      }, 2000);
    }, remaining);
    child.once("error", error => {failure = error; code = null; settle();});
    child.once("close", (exitCode, exitSignal) => {code = exitCode; signal = exitSignal; settle();});
  });
}

/** Stage complete shallow Git roots for existing offline source consumers.
 * @param {GitSourceTarget} target @param {string} git @param {string} output
 * @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{run?: typeof runGit, sourceRecipe?: typeof sourceRecipe, retryDelaysMs?: readonly number[]}} [dependencies] */
async function fetchSdkSources(target, git, output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const selected = gitSourceComponents(target), names = /** @type {(keyof typeof origins)[]} */ (Object.keys(selected));
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1" || !allowNetwork) {
    throw new Error("SDK source fetch requires Linux, MOM_AOT_BUILD_ONLY=1, and explicit --allow-network");
  }
  if (!path.isAbsolute(git) || !path.isAbsolute(output) || /[\r\n\0]/.test(git + output) ||
      names.some(name => !origins[name]) || !executableGit(git)) {
    throw new Error("SDK fetch needs an absolute executable Git and complete pinned origin cohort");
  }
  const parent = fs.realpathSync(path.dirname(output)), destination = path.join(parent, path.basename(output));
  if (fs.existsSync(destination) || (() => {try {fs.lstatSync(destination); return true;} catch {return false;}})() ||
      destination === repo || destination.startsWith(repo + path.sep) || !fs.statSync(parent).isDirectory()) {
    throw new Error("SDK source output must be a new absolute directory outside the repository");
  }
  const tool = fs.realpathSync(git), toolHash = fileHash(tool), recipeFor = dependencies.sourceRecipe || sourceRecipe;
  const recipe = recipeFor(target), fetcherHash = fileHash(__filename);
  const env = {...sdkSourceEnvironment(environment), GIT_TERMINAL_PROMPT: "0", GIT_ALLOW_PROTOCOL: "https"};
  const version = spawnSync(tool, ["--version"], {env, encoding: "utf8", timeout: 5000, maxBuffer: 4096});
  if (version.error || version.status !== 0 || !/^git version \d+\./.test(version.stdout || "")) {throw new Error("SDK fetch tool is not a working Git executable");}
  const run = dependencies.run || runGit, retryDelaysMs = dependencies.retryDelaysMs || archiveRetryDelaysMs;
  fs.mkdirSync(destination); fs.mkdirSync(path.join(destination, "roots")); fs.mkdirSync(path.join(destination, "logs"));
  const template = path.join(destination, ".empty-git-template"); fs.mkdirSync(template);
  /** @type {Record<string, string>} */ const roots = {};
  /** @type {Record<string, {origin: string, revision: string, root: string, gitDirectory: "bare", shallow: true, partialClone: false}>} */ const fetched = {};
  let next = 0, stopped = false;
  /** @type {string[]} */ const failures = [];
  /** @param {keyof typeof origins} name */
  async function one(name) {
    const component = selected[name], origin = origins[name];
    if (!component) {throw new Error("SDK source cohort changed during fetch: " + name);}
    const revision = component.revision, root = path.join(destination, "roots", name), log = path.join(destination, "logs", name + ".log"), deadline = Date.now() + 600000;
    fs.mkdirSync(root); roots[name] = root;
    const gitRun = (/** @type {string[]} */ args) => run(tool, args, env, log, deadline);
    if (fs.readdirSync(template).length) {throw new Error("Git template directory changed during SDK fetch");}
    await gitRun(["init", "--bare", `--template=${template}`, "--quiet", root]);
    await gitRun(["-C", root, "remote", "add", "origin", origin]);
    // Only the network step is retried; the fetched revision is still checked below.
    for (let attempt = 1; ; attempt++) {
      try {
        await gitRun(["-C", root, "fetch", "--depth=1", "--no-tags", "--no-recurse-submodules", "origin", revision]);
        break;
      } catch (error) {
        if (attempt >= gitFetchAttempts || Date.now() >= deadline) {throw error;}
        console.error("[native-sdk] Git fetch retry " + attempt + " after a failure: " + name);
        await sleep(retryDelaysMs[attempt - 1] ?? 0);
      }
    }
    await gitRun(["-C", root, "cat-file", "-e", revision + "^{commit}"]);
    // FETCH_HEAD alone is not cloned by Git-aware SDK producers. Keep the pinned commit
    // reachable (also safe from Git GC) and make a normal offline clone transfer it.
    await gitRun(["-C", root, "update-ref", "refs/heads/mom-native-source", revision]);
    await gitRun(["-C", root, "symbolic-ref", "HEAD", "refs/heads/mom-native-source"]);
    await gitRun(["-C", root, "fsck", "--connectivity-only", "--no-reflogs"]);
    const config = readBuildText(path.join(root, "config"), 65536), remote = config.match(/\[remote "origin"\]([\s\S]*?)(?=\n\[|$)/)?.[1];
    if (!/\[core\][\s\S]*?\bbare\s*=\s*true/i.test(config) || remote?.match(/^\s*url\s*=\s*(.*?)\s*$/m)?.[1] !== origin ||
        /partialclone|promisor/i.test(config) || !fs.existsSync(path.join(root, "shallow")) ||
        fs.readdirSync(path.join(root, "objects", "pack")).some(file => file.endsWith(".promisor"))) {
      throw new Error("Fetched SDK Git root is not complete, shallow, and non-promisor: " + name);
    }
    fetched[name] = {origin, revision, root, gitDirectory: "bare", shallow: true, partialClone: false};
  }
  async function worker() {
    while (!stopped) {
      const index = next++;
      if (index >= names.length) {return;}
      const name = names[index];
      if (!name) {return;}
      try {await one(name);} catch (error) {failures.push(error instanceof Error ? error.message : "SDK source fetch failed"); stopped = true;}
    }
  }
  await Promise.all(Array.from({length: Math.min(4, names.length)}, () => worker()));
  if (failures.length) {throw new Error(failures.join("\n"));}
  if (fileHash(tool) !== toolHash || fileHash(__filename) !== fetcherHash ||
      JSON.stringify(recipeFor(target)) !== JSON.stringify(recipe) ||
      names.some(name => fetched[name]?.revision !== selected[name]?.revision)) {
    throw new Error("Git tool, SDK source pins, or preparation recipe changed during fetch");
  }
  const provenance = {version: 1, target, status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
    git: {path: tool, sha256: toolHash}, fetcherSha256: fetcherHash,
    recipe: {sha256: createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), files: recipe.graph}, components: fetched};
  fs.writeFileSync(path.join(destination, "roots.json"), JSON.stringify(roots, null, 2) + "\n", {flag: "wx"});
  fs.writeFileSync(path.join(destination, "fetch.json"), JSON.stringify(provenance, null, 2) + "\n", {flag: "wx"});
  return provenance;
}

/** Archive leaf names must also be ordinary files on Windows, not DOS device aliases.
 * @param {unknown} value @param {number} [maximum] @returns {value is string} */
function archiveLeaf(value, maximum = 255) {
  return typeof value === "string" && value.length <= maximum && /^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(value) &&
    !/[. ]$/.test(value) && !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(value);
}

/** One bounded, no-clobber acquisition path for maintained, pinned archive cohorts.
 * This trusted-code API is not a validator for arbitrary cohort JSON: callers must supply the
 * official URLs/origin allowlist, content hashes, limits, and recipe from maintained code.
 * @param {string} output @param {NodeJS.ProcessEnv} environment @param {boolean} allowNetwork
 * @param {PinnedArchiveCohort} cohort */
async function fetchArchiveCohort(output, environment, allowNetwork, cohort) {
  const {target, sourceLabel, archiveLabel, specs, pins, maxBytes, timeoutMs, recipeFor, pinSnapshot, fetchImpl,
    retryDelaysMs = archiveRetryDelaysMs} = cohort;
  if (!["linux", "win32"].includes(process.platform) || environment["MOM_AOT_BUILD_ONLY"] !== "1" || !allowNetwork) {
    throw new Error(sourceLabel + " fetch requires Linux/Windows, MOM_AOT_BUILD_ONLY=1, and explicit --allow-network");
  }
  const outputError = sourceLabel + " output must be a new absolute directory outside the repository";
  if (typeof output !== "string" || !path.isAbsolute(output) || /[\r\n\0]/.test(output)) {throw new Error(outputError);}
  const parent = fs.realpathSync(path.dirname(output)), destination = path.join(parent, path.basename(output));
  if (destination !== path.resolve(output) || destination === repo || destination.startsWith(repo + path.sep) ||
      (() => {try {fs.lstatSync(destination); return true;} catch {return false;}})() || !fs.statSync(parent).isDirectory()) {throw new Error(outputError);}
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
    throw new Error(sourceLabel + " fetch limits are invalid");
  }
  if (typeof fetchImpl !== "function") {throw new Error(sourceLabel + " fetch needs the built-in HTTPS fetch implementation");}
  const names = Object.keys(specs).sort();
  const fileNames = names.map(name => specs[name]?.fileName);
  if (!names.length || names.join(",") !== Object.keys(pins).sort().join(",") || names.some(name =>
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(name) || !/^[0-9a-f]{64}$/.test(pins[name]?.sha256 || "") ||
    !archiveLeaf(specs[name]?.fileName) || typeof specs[name]?.url !== "string" || !specs[name].url ||
    !Array.isArray(specs[name]?.origins) || specs[name].origins.length === 0 ||
    specs[name].origins.some(origin => typeof origin !== "string"))) {
    throw new Error(archiveLabel + " fetch needs its exact pinned archive cohort");
  }
  if (new Set(fileNames.map(name => name?.toLowerCase())).size !== fileNames.length) {
    throw new Error(archiveLabel + " archive filenames must be unique safe leaf names");
  }
  const initialSpecs = JSON.stringify(specs), initialPins = pinSnapshot();
  const recipe = recipeFor(), recipeJson = JSON.stringify(recipe), fetcherHash = fileHash(__filename);
  fs.mkdirSync(destination, {mode: 0o700});
  let staging = "";
  /** @type {Array<{path: string, dev: number, ino: number}>} */ const promoted = [];
  /** @type {Record<string, {path: string, url: string, sha256: string, bytes: number}>} */ const archives = {};
  /** @param {string} file @param {number} dev @param {number} ino */
  function unlinkIfOwned(file, dev, ino) {
    try {
      const stat = fs.lstatSync(file);
      if (stat.isFile() && !stat.isSymbolicLink() && stat.dev === dev && stat.ino === ino) {fs.unlinkSync(file);}
    } catch (error) {if (Reflect.get(Object(error), "code") !== "ENOENT") {throw error;}}
  }
  try {
    staging = fs.mkdtempSync(path.join(destination, ".download-"));
    for (const name of names) {
      const pin = pins[name], spec = specs[name];
      if (!pin || !spec) {throw new Error(archiveLabel + " cohort changed during fetch: " + name);}
      const url = spec.url, parsed = new URL(url), approvedOrigins = spec.origins;
      if (parsed.protocol !== "https:" || !approvedOrigins?.includes(parsed.origin) || approvedOrigins.some(origin => {
        try {const candidate = new URL(origin); return candidate.protocol !== "https:" || candidate.origin !== origin;} catch {return true;}
      })) {throw new Error(archiveLabel + " URL is outside the approved HTTPS origins: " + name);}
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeoutMs);
      const temporary = path.join(staging, spec.fileName), finalPath = path.join(destination, spec.fileName);
      /** Download one attempt into the staging file; only network failures, 429 and 5xx are marked transient. */
      const attempt = async () => {
        let current = parsed;
        for (let redirects = 0; ; redirects++) {
          const response = await fetchImpl(current.href, {redirect: "manual", signal: controller.signal}).catch(error => {throw transient(error);});
          if (response.status >= 300 && response.status < 400) {
            const location = response.headers?.get("location");
            try {await response.body?.cancel();} catch { /* discard an untrusted redirect body */ }
            if (!location || redirects >= maxArchiveRedirects) {throw new Error(archiveLabel + " redirect is missing or excessive: " + name);}
            const next = new URL(location, current);
            if (next.protocol !== "https:" || !approvedOrigins.includes(next.origin)) {
              throw new Error(archiveLabel + " redirect left its approved HTTPS origin: " + name);
            }
            current = next;
            continue;
          }
          if (response.status !== 200 || !response.body) {
            try {await response.body?.cancel();} catch { /* discard an invalid response body */ }
            const error = new Error(archiveLabel + " download did not return a complete response: " + name + " (HTTP " + response.status + ")");
            throw response.status === 429 || response.status >= 500 ? transient(error) : error;
          }
          if (typeof response.url !== "string" || !response.url || !approvedOrigins.includes(new URL(response.url).origin)) {
            try {await response.body.cancel();} catch { /* discard an untrusted response body */ }
            throw new Error(archiveLabel + " response left its approved HTTPS origin: " + name);
          }
          const contentLength = response.headers?.get("content-length");
          if (contentLength !== null && contentLength !== undefined) {
            if (!/^(0|[1-9][0-9]*)$/.test(contentLength) || !Number.isSafeInteger(Number(contentLength))) {
              try {await response.body.cancel();} catch { /* discard a response with invalid metadata */ }
              throw new Error(archiveLabel + " has an invalid Content-Length: " + name);
            }
            if (Number(contentLength) > maxBytes) {
              try {await response.body.cancel();} catch { /* discard an oversized response */ }
              throw new Error(archiveLabel + " exceeds its configured archive bound: " + name);
            }
          }
          const hash = createHash("sha256"), oversized = new Error(archiveLabel + " exceeds its configured archive bound: " + name);
          let bytes = 0;
          const limiter = new Transform({transform(chunk, _encoding, callback) {
            bytes += chunk.length;
            if (bytes > maxBytes) {callback(oversized); return;}
            hash.update(chunk); callback(null, chunk);
          }});
          await pipeline(Readable.fromWeb(response.body), limiter, fs.createWriteStream(temporary, {flags: "wx", mode: 0o600}),
            {signal: controller.signal}).catch(error => {throw error === oversized ? error : transient(error);});
          const sha256 = hash.digest("hex");
          if (sha256 !== pin.sha256) {
            throw new Error(archiveLabel + " SHA256 differs from its pinned original: " + name +
              " (expected SHA256 " + pin.sha256 + ", actual SHA256 " + sha256 + ", downloaded " + bytes + " bytes)");
          }
          fs.chmodSync(temporary, 0o444);
          fs.linkSync(temporary, finalPath); // Atomic no-clobber promotion within the new output directory.
          const stat = fs.lstatSync(finalPath);
          promoted.push({path: finalPath, dev: stat.dev, ino: stat.ino});
          fs.unlinkSync(temporary);
          if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o222) !== 0) {throw new Error("Promoted " + archiveLabel + " is not read-only: " + name);}
          return {path: finalPath, url, sha256, bytes};
        }
      };
      try {
        for (let retry = 0; ; retry++) {
          try {
            archives[name] = await attempt();
            break;
          } catch (error) {
            fs.rmSync(temporary, {force: true});
            const delay = retryDelaysMs[retry];
            if (controller.signal.aborted || delay === undefined || !transientErrors.has(Object(error))) {throw error;}
            console.error("[native-sdk] " + archiveLabel + " retry " + (retry + 1) + " after a transient failure: " + name);
            await sleep(delay, undefined, {signal: controller.signal});
          }
        }
      } catch (error) {
        if (controller.signal.aborted) {throw new Error(archiveLabel + " component deadline exceeded: " + name, {cause: error});}
        throw error;
      } finally {clearTimeout(timer);}
    }
    if (fileHash(__filename) !== fetcherHash || JSON.stringify(specs) !== initialSpecs ||
        JSON.stringify(recipeFor()) !== recipeJson || pinSnapshot() !== initialPins) {
      throw new Error(archiveLabel + " fetcher, pins, or recipe changed during download");
    }
    const provenance = {version: 1, target, status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
      fetcherSha256: fetcherHash, recipe: {sha256: createHash("sha256").update(recipeJson).digest("hex"), files: recipe.graph}, archives};
    const receipt = path.join(staging, "fetch.json"), receiptTarget = path.join(destination, "fetch.json");
    fs.writeFileSync(receipt, JSON.stringify(provenance, null, 2) + "\n", {flag: "wx", mode: 0o600});
    fs.chmodSync(receipt, 0o444); fs.linkSync(receipt, receiptTarget);
    const receiptStat = fs.lstatSync(receiptTarget);
    promoted.push({path: receiptTarget, dev: receiptStat.dev, ino: receiptStat.ino});
    if (!receiptStat.isFile() || receiptStat.isSymbolicLink() || (receiptStat.mode & 0o222) !== 0) {throw new Error(archiveLabel + " fetch receipt is not read-only");}
    fs.rmSync(staging, {recursive: true});
    return provenance;
  } catch (error) {
    if (staging) {try {fs.rmSync(staging, {recursive: true, force: true});} catch { /* retain the original download error */ }}
    for (const file of promoted.reverse()) {
      try {unlinkIfOwned(file.path, file.dev, file.ino);} catch { /* do not mask the fetch failure */ }
    }
    try {fs.rmdirSync(destination);} catch { /* preserve anything not created by this fetch */ }
    throw error;
  }
}

/** Keep the small I/O archive cohort separate from Git-root fetching. Test seams may
 * narrow limits and supply fixture hashes but cannot alter archive names or origins.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, pins?: IoArchivePins, maxBytes?: number, timeoutMs?: number,
 *   recipe?: typeof ioSourceRecipe}} [dependencies] */
async function fetchIoArchives(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const pins = /** @type {IoArchivePins} */ (dependencies.pins || ioPins), names = /** @type {(keyof IoArchivePins)[]} */ (Object.keys(ioPins).sort());
  if (!pins || Object.keys(pins).sort().join(",") !== names.join(",") || names.some(name =>
    pins[name]?.directory !== ioPins[name]?.directory || !/^[0-9a-f]{64}$/.test(pins[name]?.sha256 || ""))) {
    throw new Error("I/O source fetch needs the exact pinned libuv/zlib archive cohort");
  }
  const maxBytes = dependencies.maxBytes ?? maxIoArchiveBytes, timeoutMs = dependencies.timeoutMs ?? ioArchiveTimeoutMs;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > maxIoArchiveBytes ||
      !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > ioArchiveTimeoutMs) {
    throw new Error("I/O source fetch limits may only be narrowed from their production bounds");
  }
  const hashes = /** @type {ArchiveHashes} */ (Object.fromEntries(names.map(name => [name, {sha256: pins[name].sha256}])));
  const catalog = ioArchiveCatalog(hashes), specs = /** @type {ArchiveSpecs} */ (Object.fromEntries(names.map(name =>
    [name, {...catalog[name], origins: ioArchiveOrigins[name]}])));
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "io", sourceLabel: "I/O source", archiveLabel: "I/O archive",
    specs, pins: hashes, maxBytes, timeoutMs, recipeFor: dependencies.recipe || ioSourceRecipe,
    pinSnapshot: () => JSON.stringify(pins), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Pinned original sources needed by the cold LLVM/OpenSSL SDK build; extraction stays separate.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes, maxBytes?: number, timeoutMs?: number,
 *   recipe?: typeof buildSourceRecipe}} [dependencies] */
async function fetchBuildArchives(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const expected = buildArchiveCatalog(dependencies.hashes), names = /** @type {BuildArchiveName[]} */ (Object.keys(expected).sort());
  const hashes = Object.fromEntries(names.map(name => [name, {sha256: expected[name].sha256}]));
  const maxBytes = dependencies.maxBytes ?? maxBuildArchiveBytes, timeoutMs = dependencies.timeoutMs ?? buildArchiveTimeoutMs;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > maxBuildArchiveBytes ||
      !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > buildArchiveTimeoutMs) {
    throw new Error("Native SDK source fetch limits may only be narrowed from their production bounds");
  }
  const specs = /** @type {ArchiveSpecs} */ (Object.fromEntries(names.map(name => [name,
    {fileName: expected[name].fileName, url: expected[name].url, origins: buildArchiveOrigins}])));
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "build", sourceLabel: "Native SDK source", archiveLabel: "Native SDK source archive",
    specs, pins: hashes, maxBytes, timeoutMs, recipeFor: dependencies.recipe || buildSourceRecipe,
    pinSnapshot: () => JSON.stringify(hashes), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Fetch the original Windows OpenSSL archive through the shared bounded cohort path.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes, timeoutMs?: number}} [dependencies] */
async function fetchWindowsOpenSslArchives(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const catalog = windowsOpenSslArchiveCatalog(dependencies.hashes), hashes = {openssl: {sha256: catalog.openssl.sha256}},
    timeoutMs = dependencies.timeoutMs ?? buildArchiveTimeoutMs;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > buildArchiveTimeoutMs) {
    throw new Error("Windows OpenSSL source fetch timeout may only be narrowed from its production bound");
  }
  const specs = {openssl: {...catalog.openssl, origins: buildArchiveOrigins}};
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "windows-openssl", sourceLabel: "Windows OpenSSL source",
    archiveLabel: "OpenSSL source archive", specs, pins: hashes, maxBytes: maxWindowsOpenSslArchiveBytes, timeoutMs,
    recipeFor: opensslBuildRecipe, pinSnapshot: () => JSON.stringify(hashes), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Acquire the producer-pinned zstd source archive; extraction/build remain separate.
 * Test seams may replace only the digest or narrow limits, never the archive identity or origins.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes, maxBytes?: number, timeoutMs?: number,
 *   recipe?: typeof zstdSourceRecipe}} [dependencies] */
async function fetchZstdArchive(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const catalog = zstdArchiveCatalog(dependencies.hashes), hashes = {zstd: {sha256: catalog["zstd"].sha256}};
  const maxBytes = dependencies.maxBytes ?? maxZstdArchiveBytes, timeoutMs = dependencies.timeoutMs ?? buildArchiveTimeoutMs;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > maxZstdArchiveBytes ||
      !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > buildArchiveTimeoutMs) {
    throw new Error("zstd source fetch limits may only be narrowed from their production bounds");
  }
  const specs = {zstd: {...catalog["zstd"], origins: buildArchiveOrigins}};
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "zstd", sourceLabel: "zstd source", archiveLabel: "zstd source archive",
    specs, pins: hashes, maxBytes, timeoutMs, recipeFor: dependencies.recipe || zstdSourceRecipe,
    pinSnapshot: () => JSON.stringify(hashes), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Acquire the pinned glibc sysroot packages from the project's mirror release only.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes}} [dependencies] */
async function fetchSysrootPackages(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const catalog = sysrootArchiveCatalog(dependencies.hashes), hashes = /** @type {ArchiveHashes} */ (Object.fromEntries(
    Object.entries(catalog).map(([name, row]) => [name, {sha256: row.sha256}])));
  const specs = /** @type {ArchiveSpecs} */ (Object.fromEntries(Object.entries(catalog).map(([name, row]) => [name, {...row, origins: linuxSysrootOrigins}])));
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "sysroot", sourceLabel: "Linux sysroot input",
    archiveLabel: "Linux sysroot package", specs, pins: hashes, maxBytes: maxSysrootArchiveBytes, timeoutMs: buildArchiveTimeoutMs,
    recipeFor: sysrootSourceRecipe, pinSnapshot: () => JSON.stringify(hashes), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Acquire the pinned upstream libdrm release archive from the same mirror release.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes}} [dependencies] */
async function fetchLibdrmArchive(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const catalog = libdrmArchiveCatalog(dependencies.hashes), hashes = {libdrm: {sha256: catalog.libdrm.sha256}};
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "libdrm", sourceLabel: "libdrm source", archiveLabel: "libdrm source archive",
    specs: {libdrm: {...catalog.libdrm, origins: linuxSysrootOrigins}}, pins: hashes, maxBytes: maxSysrootArchiveBytes,
    timeoutMs: buildArchiveTimeoutMs, recipeFor: libdrmBuildRecipe, pinSnapshot: () => JSON.stringify(hashes),
    fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Acquire only the selected vendor compiler archive cohort, using the existing bounded
 * downloader. This neither installs an SDK nor reruns/restamps existing compiler proofs.
 * @param {"linux" | "windows"} platform @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{cohort?: "linux" | "windows" | "linux-12.6", fetch?: IoArchiveFetch, hashes?: ArchiveHashes, maxBytes?: number, timeoutMs?: number,
 *   recipe?: typeof nvrtcSourceRecipe}} [dependencies] */
async function fetchNvrtcArchives(platform, output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const cohort = dependencies.cohort === undefined ? platform : dependencies.cohort, catalog = nvrtcArchiveCatalog(platform, dependencies.hashes, cohort);
  const hashes = Object.fromEntries(Object.entries(catalog).map(([name, item]) => [name, {sha256: item.sha256}]));
  const maxBytes = dependencies.maxBytes ?? maxNvrtcArchiveBytes, timeoutMs = dependencies.timeoutMs ?? buildArchiveTimeoutMs;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > maxNvrtcArchiveBytes ||
      !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > buildArchiveTimeoutMs) {
    throw new Error("NVRTC vendor input limits may only be narrowed from their production bounds");
  }
  const specs = Object.fromEntries(Object.entries(catalog).map(([name, item]) => [name, {...item, origins: [nvrtcArchiveOrigin]}]));
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "nvrtc-" + cohort, sourceLabel: "NVRTC vendor input",
    archiveLabel: "NVRTC vendor archive", specs, pins: hashes, maxBytes, timeoutMs, recipeFor: dependencies.recipe || nvrtcSourceRecipe,
    pinSnapshot: () => JSON.stringify(hashes), fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Keep OS-specific compiler pins distinct. Only fixture digests may be substituted;
 * archive names and official URLs are always canonical. The probe is lazy-loaded so
 * ordinary source fetches do not acquire the native runtime dependency graph.
 * @param {"linux" | "windows"} platform @param {ArchiveHashes} [hashes] @param {"linux" | "windows" | "linux-12.6"} [cohort]
 * @returns {Record<string, {fileName: string, sha256: string, url: string}>} */
function nvrtcArchiveCatalog(platform, hashes, cohort = platform) {
  if (platform !== "linux" && platform !== "windows") {throw new Error("NVRTC vendor input target must be linux or windows");}
  const {nvrtcCohortInfo} = require("./probe-static-nvrtc"), pins = nvrtcCohortInfo(platform, cohort), names = Object.keys(pins.archives).sort();
  if (hashes && (Object.keys(hashes).sort().join(",") !== names.join(",") ||
      names.some(name => !/^[0-9a-f]{64}$/.test(hashes[name]?.sha256 || "")))) {
    throw new Error("NVRTC vendor fetch needs the exact pinned compiler archive cohort");
  }
  return Object.fromEntries(Object.entries(pins.archives).map(([name, pin]) => [name, {...pin, sha256: hashes?.[name]?.sha256 || pin.sha256,
    url: `${nvrtcArchiveOrigin}/compute/cuda/redist/${name === "nvrtc" ? "cuda_nvrtc" : "libnvptxcompiler"}/${platform}-x86_64/${pin.fileName}`} ]));
}

/** Canonical source archive cohort; test seams may replace digests but not names or URLs.
 * @param {ArchiveHashes} [hashes] @returns {Record<BuildArchiveName, {fileName: string, sha256: string, url: string}>} */
function buildArchiveCatalog(hashes) {
  const canonical = /** @type {Record<BuildArchiveName, {fileName: string, sha256: string, url: string}>} */ ({
    llvm: {fileName: llvmArchiveName, sha256: llvmArchiveSha256, url: buildArchiveUrls.llvm},
    openssl: {fileName: "openssl-" + opensslPin.version + ".tar.gz", sha256: opensslPin.sha256, url: buildArchiveUrls.openssl},
  });
  if (!hashes) {return canonical;}
  const names = /** @type {BuildArchiveName[]} */ (Object.keys(canonical).sort());
  if (Object.keys(hashes).sort().join(",") !== names.join(",") || names.some(name => !/^[0-9a-f]{64}$/.test(hashes[name]?.sha256 || ""))) {
    throw new Error("Native SDK source fetch needs the exact pinned LLVM/OpenSSL archive cohort");
  }
  const result = Object.fromEntries(names.map(name => [name, {...canonical[name], sha256: hashes[name]?.sha256 || ""}]));
  return /** @type {Record<BuildArchiveName, {fileName: string, sha256: string, url: string}>} */ (result);
}

/** Preserve the original Windows OpenSSL cohort while allowing fixture-only digest substitution.
 * @param {ArchiveHashes} [hashes] */
function windowsOpenSslArchiveCatalog(hashes) {
  if (hashes && (Object.keys(hashes).join(",") !== "openssl" || !/^[0-9a-f]{64}$/.test(hashes["openssl"]?.sha256 || ""))) {
    throw new Error("Windows OpenSSL source fetch needs its exact pinned archive cohort");
  }
  return {openssl: {fileName: "openssl-" + opensslPin.version + ".tar.gz", sha256: hashes?.["openssl"]?.sha256 || opensslPin.sha256,
    url: buildArchiveUrls.openssl}};
}

/** Share the canonical I/O names and origins between acquisition and offline admission.
 * Fixture hashes may differ, but never the cohort, filenames or origins.
 * @param {ArchiveHashes} [hashes] */
function ioArchiveCatalog(hashes) {
  const names = /** @type {(keyof IoArchivePins)[]} */ (Object.keys(ioPins).sort());
  if (hashes && (Object.keys(hashes).sort().join(",") !== names.join(",") ||
      names.some(name => !/^[0-9a-f]{64}$/.test(hashes[name]?.sha256 || "")))) {
    throw new Error("I/O source fetch needs the exact pinned libuv/zlib archive cohort");
  }
  return Object.fromEntries(names.map(name => [name, {fileName: ioPins[name].directory + ".tar.gz",
    sha256: hashes?.[name]?.sha256 || ioPins[name].sha256, url: ioArchiveUrls[name]}]));
}

/** All archive cohorts use the same bounded receipt, identity and race checks. Keep this
 * with acquisition rather than duplicating its contract in each CI consumer.
 * @param {string} output @param {string} target
 * @param {Record<string, {fileName: string, sha256: string, url: string}>} catalog
 * @param {typeof buildSourceRecipe} recipeFor @param {number} maximum
 * @returns {BuildArchiveReceipt} */
function verifyArchiveCohort(output, target, catalog, recipeFor, maximum) {
  if (typeof output !== "string" || !path.isAbsolute(output) || /[\r\n\0]/.test(output)) {
    throw new Error("Native SDK source verification needs an absolute output directory");
  }
  const directory = path.resolve(output), directoryStat = fs.lstatSync(directory, {bigint: true});
  if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {throw new Error("Native SDK source output must be a real directory");}
  const canonicalDirectory = fs.realpathSync(directory);
  if (canonicalDirectory !== directory || directory === repo || directory.startsWith(repo + path.sep)) {
    throw new Error("Native SDK source output must be canonical and outside the repository");
  }
  const entries = Object.entries(catalog).sort(([left], [right]) => left.localeCompare(right)), names = entries.map(([name]) => name);
  const expectedFiles = ["fetch.json", ...entries.map(([, item]) => item.fileName)].sort();
  if (!isDeepStrictEqual(fs.readdirSync(directory).sort(), expectedFiles)) {
    throw new Error("Native SDK source output must contain only its receipt and pinned archive cohort");
  }

  const receiptPath = path.join(directory, "fetch.json"), verifiedReceipt = readReadonlyArchiveReceipt(receiptPath), receipt = verifiedReceipt.receipt;
  if (!receipt || receipt.version !== 1 || receipt.target !== target || receipt.status !== "PASS" || receipt.networked !== true ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.publicationQualified !== false) {
    throw new Error("Native SDK source receipt is not a valid build-only, unpublished acquisition");
  }
  // This records the historical downloader, not archive identity. Unrelated acquisition changes
  // must not invalidate immutable sources: current URLs, pins, recipe and actual bytes are checked
  // below. Preserve the original receipt; its self-reported provenance is not authentication.
  if (typeof receipt.fetcherSha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.fetcherSha256)) {
    throw new Error("Native SDK source receipt has invalid fetcher provenance");
  }
  // Builder fingerprints describe acquisition history, not immutable archive identity.
  // Current URLs/pins and actual bytes below are authoritative; compiled SDK receipts
  // separately require current build recipes. Validate metadata without restamping it.
  const recipe = recipeFor(), graph = receipt.recipe?.files;
  if (!graph || typeof graph !== "object" || Array.isArray(graph) ||
      !isDeepStrictEqual(Object.keys(graph).sort(), Object.keys(recipe.graph).sort()) ||
      Object.values(graph).some(value => typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) ||
      receipt.recipe.sha256 !== createHash("sha256").update(JSON.stringify({files: {}, graph})).digest("hex")) {
    throw new Error("Native SDK source receipt has stale builder recipe provenance");
  }
  if (!receipt.archives || Object.keys(receipt.archives).sort().join(",") !== names.join(",")) {
    throw new Error("Native SDK source receipt has an incomplete archive cohort");
  }

  /** @type {[string, import("node:fs").BigIntStats, string][]} */
  const snapshots = [[receiptPath, verifiedReceipt.snapshot, "fetch receipt"]];
  for (const [name, expected] of entries) {
    const archive = receipt.archives[name], file = path.join(canonicalDirectory, expected.fileName);
    const before = readonlyArchiveStat(file, name, maximum);
    if (!archive || archive.path !== file || archive.url !== expected.url || archive.sha256 !== expected.sha256 ||
        !Number.isSafeInteger(archive.bytes) || BigInt(archive.bytes) !== before.size) {
      throw new Error("Native SDK source receipt metadata does not match the pinned " + name + " archive");
    }
    if (fileHash(file) !== expected.sha256) {throw new Error("Native SDK source archive digest differs from its pin: " + name);}
    const after = readonlyArchiveStat(file, name, maximum);
    if (!sameFileIdentity(before, after)) {
      throw new Error("Native SDK source archive changed during verification: " + name);
    }
    snapshots.push([file, before, name]);
  }
  for (const [file, snapshot, label] of snapshots) {
    if (!sameFileIdentity(snapshot, readonlyArchiveStat(file, label, maximum))) {
      throw new Error("Native SDK source " + label + " changed during verification");
    }
  }
  const directoryAfter = fs.lstatSync(directory, {bigint: true});
  if (!directoryAfter.isDirectory() || directoryAfter.isSymbolicLink() || !sameFileIdentity(directoryStat, directoryAfter) ||
      !isDeepStrictEqual(fs.readdirSync(directory).sort(), expectedFiles)) {
    throw new Error("Native SDK source output changed during verification");
  }
  return receipt;
}

/** Check original LLVM/OpenSSL inputs without network access or mutation.
 * @param {string} output @param {{hashes?: ArchiveHashes, recipe?: typeof buildSourceRecipe}} [dependencies] */
function verifyBuildArchives(output, dependencies = {}) {
  return verifyArchiveCohort(output, "build", buildArchiveCatalog(dependencies.hashes), dependencies.recipe || buildSourceRecipe, maxBuildArchiveBytes);
}

/** Verify the original Windows OpenSSL acquisition without network or mutation.
 * @param {string} output @param {{hashes?: ArchiveHashes}} [dependencies] */
function verifyWindowsOpenSslArchives(output, dependencies = {}) {
  return verifyArchiveCohort(output, "windows-openssl", windowsOpenSslArchiveCatalog(dependencies.hashes),
    opensslBuildRecipe, maxWindowsOpenSslArchiveBytes);
}

/** Check original libuv/zlib inputs through the same offline admission as LLVM/OpenSSL.
 * @param {string} output @param {{hashes?: ArchiveHashes, recipe?: typeof ioSourceRecipe}} [dependencies] */
function verifyIoArchives(output, dependencies = {}) {
  return verifyArchiveCohort(output, "io", ioArchiveCatalog(dependencies.hashes), dependencies.recipe || ioSourceRecipe, maxIoArchiveBytes);
}

/** Check the original zstd archive without network access, extraction, or compilation.
 * @param {string} output @param {{hashes?: ArchiveHashes, recipe?: typeof zstdSourceRecipe}} [dependencies] */
function verifyZstdArchive(output, dependencies = {}) {
  return verifyArchiveCohort(output, "zstd", zstdArchiveCatalog(dependencies.hashes), dependencies.recipe || zstdSourceRecipe, maxZstdArchiveBytes);
}

/** Check the sysroot packages offline; extraction and admission belong to native-linux-sysroot.js.
 * @param {string} output @param {{hashes?: ArchiveHashes}} [dependencies] */
function verifySysrootPackages(output, dependencies = {}) {
  return verifyArchiveCohort(output, "sysroot", sysrootArchiveCatalog(dependencies.hashes), sysrootSourceRecipe, maxSysrootArchiveBytes);
}

/** Check the libdrm source archive offline. @param {string} output @param {{hashes?: ArchiveHashes}} [dependencies] */
function verifyLibdrmArchive(output, dependencies = {}) {
  return verifyArchiveCohort(output, "libdrm", libdrmArchiveCatalog(dependencies.hashes), libdrmBuildRecipe, maxSysrootArchiveBytes);
}

/** Read-only original-input admission, not SDK build or redistribution qualification.
 * @param {"linux" | "windows"} platform @param {string} output
 * @param {{cohort?: "linux" | "windows" | "linux-12.6", hashes?: ArchiveHashes, recipe?: typeof nvrtcSourceRecipe}} [dependencies] */
function verifyNvrtcArchives(platform, output, dependencies = {}) {
  const cohort = dependencies.cohort === undefined ? platform : dependencies.cohort;
  return verifyArchiveCohort(output, "nvrtc-" + cohort, nvrtcArchiveCatalog(platform, dependencies.hashes, cohort), dependencies.recipe || nvrtcSourceRecipe, maxNvrtcArchiveBytes);
}

/** A Windows host-SDK archive is a payload selected from the pinned VS catalog. */
/** @typedef {{id: string, version: string, fileName: string, vendorFileName: string, url: string,
 * sha256: string, format: "vsix" | "msi", declaredBytes: number, maxBytes: number}} WindowsHostPackage */
/** @typedef {{id: string, version: string, fileName: string, vendorFileName: string, url: string, sha256: string,
 * format: "cab", declaredBytes: number, maxBytes: number, msiKey: string, diskId: string}} WindowsHostCabinet */
/** @typedef {{version: 1, target: "windows-host", sourceCatalog: {sha256: string, bytes: number, productBuild: string,
 * productDisplayVersion: string, publisherAuthenticated: false, channelCatalogBindingVerified: false}, roots: string[],
 * packages: Record<string, WindowsHostPackage>, selections: Array<{package: string, sourcePrefix: string,
 * destinationPrefix: string, kind: "tree" | "file"}>, msiCabinets: Record<string, Array<{diskId: string, cabinet: string}>>,
 * cabinets: Record<string, WindowsHostCabinet>}} WindowsHostSdkLock */

/** @template T @param {T} value @returns {T} */
function freezeTree(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) {return value;}
  for (const item of Object.values(/** @type {Record<string, unknown>} */ (value))) {freezeTree(item);}
  return /** @type {T} */ (Object.freeze(value));
}

/** @param {string} value */
function safeRelativeSdkPath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 512 && !/[\\\r\n\0:]/.test(value) &&
    !value.startsWith("/") && value.split("/").every(part => part && part !== "." && part !== "..");
}

/** Validate the maintained lock before it can drive URLs, receipt names or output paths. */
function validateWindowsHostSdkLock() {
  const lock = /** @type {WindowsHostSdkLock} */ (windowsHostLock);
  const expectedRoots = ["sdk/include/ucrt", "sdk/include/shared", "sdk/include/um", "sdk/include/winrt",
    "sdk/lib/ucrt", "sdk/lib/um", "msvc/include", "msvc/lib"];
  const lockHash = lock?.sourceCatalog?.sha256;
  if (!lock || lock.version !== 1 || lock.target !== "windows-host" || !/^[0-9a-f]{64}$/.test(lockHash || "") ||
      !Number.isSafeInteger(lock.sourceCatalog.bytes) || lock.sourceCatalog.bytes < 1 ||
      lock.sourceCatalog.publisherAuthenticated !== false || lock.sourceCatalog.channelCatalogBindingVerified !== false ||
      !isDeepStrictEqual(lock.roots, expectedRoots) || !lock.packages || !lock.cabinets || !Array.isArray(lock.selections) || !lock.msiCabinets) {
    throw new Error("Windows host-SDK lock has invalid metadata or source-root scope");
  }
  const localNames = new Set(), packageKeys = Object.keys(lock.packages), cabinetKeys = Object.keys(lock.cabinets);
  /** @param {string} key @param {WindowsHostPackage | WindowsHostCabinet} row @param {"vsix"|"msi"|"cab"} format */
  function validateArchive(key, row, format) {
    const vendor = row.vendorFileName?.replace(/\\/g, "/");
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(key) || row.format !== format ||
        !archiveLeaf(row.fileName, 128) ||
        !vendor || vendor.startsWith("/") || vendor.split("/").some(part => !part || part === "." || part === ".." || part.includes(":")) ||
        typeof row.id !== "string" || !row.id || typeof row.version !== "string" || !row.version ||
        !/^[0-9a-f]{64}$/.test(row.sha256) || !Number.isSafeInteger(row.declaredBytes) || row.declaredBytes < 1 ||
        row.maxBytes !== maxWindowsHostArchiveBytes || row.declaredBytes > row.maxBytes) {
      throw new Error("Windows host-SDK archive lock row is malformed: " + key);
    }
    const url = new URL(row.url);
    if (url.protocol !== "https:" || url.origin !== windowsHostOrigin || url.username || url.password || url.search || url.hash) {
      throw new Error("Windows host-SDK archive URL is outside the pinned vendor origin: " + key);
    }
    const localName = row.fileName.toLowerCase();
    if (localNames.has(localName)) {throw new Error("Windows host-SDK archive filename collides: " + row.fileName);}
    localNames.add(localName);
  }
  if (!packageKeys.length || !cabinetKeys.length) {throw new Error("Windows host-SDK archive lock is empty");}
  for (const [key, row] of Object.entries(lock.packages)) {
    if (!row || (row.format !== "vsix" && row.format !== "msi")) {throw new Error("Windows host-SDK package format is invalid: " + key);}
    validateArchive(key, row, row.format);
  }
  const selectedPackages = new Set();
  const selectedRoots = new Set();
  const selectionKeys = new Set();
  for (const selection of lock.selections) {
    if (!selection || !lock.packages[selection.package] || !safeRelativeSdkPath(selection.sourcePrefix) ||
        !safeRelativeSdkPath(selection.destinationPrefix) || (selection.kind !== "tree" && selection.kind !== "file") ||
        !expectedRoots.some(root => selection.destinationPrefix === root || selection.destinationPrefix.startsWith(root + "/"))) {
      throw new Error("Windows host-SDK package selection escapes its fixed source roots");
    }
    const signature = [selection.package, selection.sourcePrefix, selection.destinationPrefix, selection.kind].join("\0");
    if (selectionKeys.has(signature)) {throw new Error("Windows host-SDK package selection is duplicated");}
    selectionKeys.add(signature); selectedPackages.add(selection.package);
    selectedRoots.add(expectedRoots.find(root => selection.destinationPrefix === root || selection.destinationPrefix.startsWith(root + "/")));
  }
  if (packageKeys.some(key => !selectedPackages.has(key)) || expectedRoots.some(root => !selectedRoots.has(root))) {
    throw new Error("Windows host-SDK lock selections do not cover every selected package/root");
  }
  const msiKeys = packageKeys.filter(key => lock.packages[key]?.format === "msi").sort();
  if (!isDeepStrictEqual(Object.keys(lock.msiCabinets).sort(), msiKeys)) {
    throw new Error("Windows host-SDK lock has an incomplete MSI-to-CAB mapping");
  }
  const relations = new Map();
  for (const msiKey of msiKeys) {
    const msi = lock.packages[msiKey], mediaRows = lock.msiCabinets[msiKey];
    if (!msi || !mediaRows) {throw new Error("Windows host-SDK MSI package mapping is incomplete: " + msiKey);}
    const seenDisks = new Set();
    for (const relation of mediaRows) {
      if (!relation || typeof relation.diskId !== "string" || !/^\d+$/.test(relation.diskId) || seenDisks.has(relation.diskId) ||
          !lock.cabinets[relation.cabinet]) {throw new Error("Windows host-SDK MSI Media relation is malformed: " + msiKey);}
      seenDisks.add(relation.diskId);
      const key = relation.cabinet, cabinet = lock.cabinets[key];
      if (!cabinet || cabinet.msiKey !== msiKey || cabinet.diskId !== relation.diskId || cabinet.id !== msi.id ||
          cabinet.version !== msi.version) {throw new Error("Windows host-SDK CAB does not match its pinned MSI Media row: " + key);}
      relations.set(key, (relations.get(key) || 0) + 1);
    }
    if (!seenDisks.size) {throw new Error("Windows host-SDK MSI has no pinned CAB Media rows: " + msiKey);}
  }
  for (const [key, row] of Object.entries(lock.cabinets)) {
    validateArchive(key, row, "cab");
    if (relations.get(key) !== 1) {throw new Error("Windows host-SDK CAB is missing or multiply linked to MSI Media: " + key);}
  }
  return lock;
}

/** Return the immutable package/member selector lock (not an SDK admission receipt).
 * @returns {Readonly<WindowsHostSdkLock>} */
function windowsHostSdkCatalog() {
  return freezeTree(validateWindowsHostSdkLock());
}

/** Flatten only the fixed package and CAB rows into the generic immutable-archive contract.
 * Test seams may replace digests only; filenames, URLs and origins stay lock-owned.
 * @param {ArchiveHashes} [hashes]
 * @returns {Record<string, {fileName: string, sha256: string, url: string}>} */
function windowsHostSdkArchiveCatalog(hashes) {
  const lock = windowsHostSdkCatalog(), rows = new Map();
  for (const [key, row] of Object.entries(lock.packages)) {rows.set("package." + key, row);}
  for (const [key, row] of Object.entries(lock.cabinets)) {rows.set("cab." + key, row);}
  const names = [...rows.keys()].sort();
  if (hashes && (Object.keys(hashes).sort().join(",") !== names.join(",") ||
      names.some(name => !/^[0-9a-f]{64}$/.test(hashes[name]?.sha256 || "")))) {
    throw new Error("Windows host-SDK fetch needs the exact pinned package/CAB cohort");
  }
  return Object.fromEntries(names.map(key => {
    const row = rows.get(key);
    if (!row) {throw new Error("Windows host-SDK archive lock changed while building its catalog");}
    return [key, {fileName: row.fileName, url: row.url, sha256: hashes?.[key]?.sha256 || row.sha256}];
  }));
}

/** Bind this acquisition lock, not its later extraction/build consumers. */
function windowsHostSdkSourceRecipe() {
  return sourceInputs(repo, [], ["runtime/windows-host-sdk.json", "scripts/build-source-inputs.js"]);
}

/** Fetch the fixed Windows host-SDK archive cohort; this does not extract/install/admit an SDK.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] @param {boolean} [allowNetwork]
 * @param {{fetch?: IoArchiveFetch, hashes?: ArchiveHashes}} [dependencies] */
async function fetchWindowsHostSdkArchives(output, environment = process.env, allowNetwork = false, dependencies = {}) {
  const catalog = windowsHostSdkArchiveCatalog(dependencies.hashes), hashes = /** @type {ArchiveHashes} */ (Object.fromEntries(
    Object.entries(catalog).map(([name, row]) => [name, {sha256: row.sha256}])));
  const specs = /** @type {ArchiveSpecs} */ (Object.fromEntries(Object.entries(catalog).map(([name, row]) =>
    [name, {...row, origins: [windowsHostOrigin]}])));
  return fetchArchiveCohort(output, environment, allowNetwork, {target: "windows-host", sourceLabel: "Windows host-SDK input",
    archiveLabel: "Windows host-SDK payload", specs, pins: hashes, maxBytes: maxWindowsHostArchiveBytes,
    timeoutMs: buildArchiveTimeoutMs, recipeFor: windowsHostSdkSourceRecipe, pinSnapshot: () => JSON.stringify(hashes),
    fetchImpl: dependencies.fetch || globalThis.fetch});
}

/** Verify the acquired Windows host-SDK package/CAB bytes without network, extraction, or admission.
 * @param {string} output @param {{hashes?: ArchiveHashes}} [dependencies] */
function verifyWindowsHostSdkArchives(output, dependencies = {}) {
  const catalog = windowsHostSdkArchiveCatalog(dependencies.hashes);
  return verifyArchiveCohort(output, "windows-host", catalog, windowsHostSdkSourceRecipe, maxWindowsHostArchiveBytes);
}

/** @param {string} file @param {string} label @param {number} maximum @returns {import("node:fs").BigIntStats} */
function readonlyArchiveStat(file, label, maximum) {
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1n || (stat.mode & 0o222n) !== 0n || stat.size < 1n || stat.size > BigInt(maximum)) {
    throw new Error("Native SDK source " + label + " must be a bounded, read-only, single-link regular file");
  }
  return stat;
}

/** @param {import("node:fs").BigIntStats} first @param {import("node:fs").BigIntStats} second */
function sameFileIdentity(first, second) {
  return first.dev === second.dev && first.ino === second.ino && first.size === second.size && first.mode === second.mode &&
    first.nlink === second.nlink && first.mtimeNs === second.mtimeNs && first.ctimeNs === second.ctimeNs;
}

/** Read the small receipt through a no-follow descriptor and reject link or metadata races.
 * @param {string} file @returns {{receipt: BuildArchiveReceipt, snapshot: import("node:fs").BigIntStats}} */
function readReadonlyArchiveReceipt(file) {
  const pathStat = readonlyArchiveStat(file, "fetch receipt", 1 << 20), flags = fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0);
  const fd = fs.openSync(file, flags);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || !sameFileIdentity(pathStat, before) || before.size < 1n || before.size > (1n << 20n)) {
      throw new Error("Native SDK source fetch receipt changed before reading");
    }
    const data = Buffer.alloc(Number(before.size) + 1); let size = 0;
    for (let count; size < data.length && (count = fs.readSync(fd, data, size, data.length - size, null));) {size += count;}
    const after = fs.fstatSync(fd, {bigint: true}), pathAfter = readonlyArchiveStat(file, "fetch receipt", 1 << 20);
    if (BigInt(size) !== before.size || !sameFileIdentity(before, after) || !sameFileIdentity(before, pathAfter)) {
      throw new Error("Native SDK source fetch receipt changed during verification");
    }
    return {receipt: JSON.parse(data.subarray(0, size).toString("utf8")), snapshot: before};
  } finally {fs.closeSync(fd);}
}

/** Archive mode's recipe binds the external pins to this acquisition contract. */
function ioSourceRecipe() {
  return sourceInputs(repo, [], ["scripts/build-native-io.js", "scripts/build-source-inputs.js"]);
}

/** Bind source-pin definitions, not downstream LLVM build policy. Compiled SDK receipts
 * track their own full recipes; Windows configure changes cannot invalidate Linux sources. */
function buildSourceRecipe() {
  return sourceInputs(repo, [], ["scripts/native-llvm-sdk.js", "scripts/build-native-openssl.js", "scripts/build-source-inputs.js"]);
}

/** Bind vendor pins and the existing compiler proof recipe; no claim of source rebuild. */
function nvrtcSourceRecipe() {
  return sourceInputs(repo, [], ["scripts/probe-static-nvrtc.js", "scripts/build-source-inputs.js"]);
}

/** Bind the original zstd source archive to the same producer recipe used by SDK admission. */
function zstdSourceRecipe() {
  return zstdBuildRecipe();
}

/** Bind the sysroot pins, not the producers that later compile against the prepared tree. */
function sysrootSourceRecipe() {
  return sourceInputs(repo, [], ["scripts/native-linux-sysroot.js", "scripts/build-source-inputs.js"]);
}

/** Package files keep their mirrored asset names, keyed by the asset's package stem (libstdcxx-14-dev);
 * fixture seams replace only digests. @param {ArchiveHashes} [hashes]
 * @returns {Record<string, {fileName: string, sha256: string, url: string}>} */
function sysrootArchiveCatalog(hashes) {
  const rows = linuxSysrootPackages.map(row => ({...row, key: row.asset.split("_")[0] || ""})), names = rows.map(row => row.key).sort();
  if (hashes && (Object.keys(hashes).sort().join(",") !== names.join(",") || names.some(name => !/^[0-9a-f]{64}$/.test(hashes[name]?.sha256 || "")))) {
    throw new Error("Linux sysroot fetch needs the exact pinned package cohort");
  }
  return Object.fromEntries(rows.map(row => [row.key, {fileName: row.asset,
    sha256: hashes?.[row.key]?.sha256 || row.sha256, url: linuxSysrootRelease + row.asset}]));
}

/** @param {ArchiveHashes} [hashes] */
function libdrmArchiveCatalog(hashes) {
  if (hashes && (Object.keys(hashes).join(",") !== "libdrm" || !/^[0-9a-f]{64}$/.test(hashes["libdrm"]?.sha256 || ""))) {
    throw new Error("libdrm source fetch needs the exact pinned archive cohort");
  }
  return {libdrm: {fileName: libdrmPin.fileName, sha256: hashes?.["libdrm"]?.sha256 || libdrmPin.sha256, url: libdrmPin.url}};
}

/** Keep the one-source archive's public name and release URL fixed; fixture seams replace only its digest.
 * @param {ArchiveHashes} [hashes] */
function zstdArchiveCatalog(hashes) {
  if (hashes && (Object.keys(hashes).join(",") !== "zstd" || !/^[0-9a-f]{64}$/.test(hashes["zstd"]?.sha256 || ""))) {
    throw new Error("zstd source fetch needs the exact pinned archive cohort");
  }
  return {zstd: {fileName: "zstd-" + zstdPin.version + ".tar.gz", sha256: hashes?.["zstd"]?.sha256 || zstdPin.sha256, url: zstdArchiveUrl}};
}

if (require.main === module) {
  (async () => {
    try {
      const [target, first, second, permission, extra] = process.argv.slice(2);
      if (/^(verify-)?nvrtc-(linux-12\.6|linux|windows)$/.test(target || "")) {
        const cohort = /** @type {"linux" | "windows" | "linux-12.6"} */ (target?.replace(/^verify-/, "").slice("nvrtc-".length) || ""),
          platform = /** @type {"linux" | "windows"} */ (cohort === "linux-12.6" ? "linux" : cohort);
        if (target?.startsWith("verify-")) {
          if (!first || second || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js verify-nvrtc-linux|verify-nvrtc-linux-12.6|verify-nvrtc-windows /absolute/vendor-output");}
          verifyNvrtcArchives(platform, first, {cohort});
          console.log("[native-sdk] Verified pinned NVRTC vendor archives offline; not SDK build or redistribution qualification.");
        } else {
          if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js nvrtc-linux|nvrtc-linux-12.6|nvrtc-windows /new/absolute/output --allow-network");}
          await fetchNvrtcArchives(platform, first, process.env, true, {cohort});
          console.log("[native-sdk] Fetched pinned vendor-prebuilt NVRTC inputs; no extraction, compilation or publication qualification.");
        }
      } else if (target === "verify-build" || target === "verify-io") {
        if (!first || second || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js verify-build|verify-io /absolute/source-output");}
        (target === "verify-build" ? verifyBuildArchives : verifyIoArchives)(first);
        console.log("[native-sdk] Verified pinned " + (target === "verify-build" ? "LLVM/OpenSSL" : "libuv/zlib") + " archives and source receipt offline; no extraction or compilation.");
      } else if (target === "verify-sysroot" || target === "verify-libdrm") {
        if (!first || second || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js verify-sysroot|verify-libdrm /absolute/source-output");}
        (target === "verify-sysroot" ? verifySysrootPackages : verifyLibdrmArchive)(first);
        console.log("[native-sdk] Verified pinned " + (target === "verify-sysroot" ? "glibc sysroot packages" : "libdrm source archive") + " offline; no extraction or compilation.");
      } else if (target === "sysroot" || target === "libdrm") {
        if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js sysroot|libdrm /new/absolute/output --allow-network");}
        await (target === "sysroot" ? fetchSysrootPackages : fetchLibdrmArchive)(first, process.env, true);
        console.log("[native-sdk] Fetched pinned " + (target === "sysroot" ? "glibc sysroot packages" : "libdrm source") + " from the mirror release; no extraction or compilation.");
      } else if (target === "verify-zstd") {
        if (!first || second || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js verify-zstd /absolute/source-output");}
        verifyZstdArchive(first);
        console.log("[native-sdk] Verified pinned zstd source archive and receipt offline; no extraction or compilation.");
      } else if (target === "verify-windows-host-sdk") {
        if (!first || second || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js verify-windows-host-sdk /absolute/archive-output");}
        verifyWindowsHostSdkArchives(first);
        console.log("[native-sdk] Verified pinned Windows host-SDK package/CAB bytes offline; no extraction or SDK admission.");
      } else if (target === "windows-host-sdk") {
        if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js windows-host-sdk /new/absolute/output --allow-network");}
        await fetchWindowsHostSdkArchives(first, process.env, true);
        console.log("[native-sdk] Fetched pinned Windows host-SDK packages/CABs; no extraction, installation, or SDK admission.");
      } else if (target === "build") {
        if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js build /new/absolute/output --allow-network");}
        await fetchBuildArchives(first, process.env, true);
        console.log("[native-sdk] Fetched pinned LLVM/OpenSSL source archives; no extraction or compilation.");
      } else if (target === "io") {
        if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js io /new/absolute/output --allow-network");}
        await fetchIoArchives(first, process.env, true);
        console.log("[native-sdk] Fetched pinned libuv/zlib archives; no extraction or compilation.");
      } else if (target === "zstd") {
        if (!first || second !== "--allow-network" || permission || extra) {throw new Error("Usage: fetch-native-sdk-sources.js zstd /new/absolute/output --allow-network");}
        await fetchZstdArchive(first, process.env, true);
        console.log("[native-sdk] Fetched the original pinned zstd source archive; no extraction or compilation.");
      } else {
        if (!target || !["linux", "windows", "compiler", "hermes", "acpp", "rocm"].includes(target) || !first || !second || permission !== "--allow-network" || extra) {throw new Error("Usage: fetch-native-sdk-sources.js linux|windows|compiler|hermes|acpp|rocm /absolute/git /new/absolute/output --allow-network | build|io|zstd|sysroot|libdrm|windows-host-sdk /new/absolute/output --allow-network | verify-build|verify-io|verify-zstd|verify-sysroot|verify-libdrm|verify-windows-host-sdk /absolute/source-output");}
        await fetchSdkSources(/** @type {GitSourceTarget} */ (target), first, second, process.env, true);
        console.log(`[native-sdk] Fetched original ${target} Git roots; no source extraction or compilation.`);
      }
    } catch (error) {console.error(error instanceof Error ? error.message : "SDK source fetch failed"); process.exitCode = 1;}
  })();
}

module.exports = {fetchSdkSources, fetchArchiveCohort, fetchIoArchives, fetchBuildArchives, fetchWindowsOpenSslArchives, fetchNvrtcArchives, fetchWindowsHostSdkArchives,
  verifyBuildArchives, verifyWindowsOpenSslArchives, verifyIoArchives, verifyNvrtcArchives, verifyWindowsHostSdkArchives,
  ioArchiveUrls, buildArchiveUrls, nvrtcArchiveCatalog, zstdArchiveCatalog, origins, runGit, gitSourceComponents, sourceRecipe, ioSourceRecipe, buildSourceRecipe,
  nvrtcSourceRecipe, zstdSourceRecipe, fetchZstdArchive, verifyZstdArchive, windowsHostSdkCatalog, windowsHostSdkArchiveCatalog, windowsHostSdkSourceRecipe,
  fetchSysrootPackages, verifySysrootPackages, sysrootArchiveCatalog, sysrootSourceRecipe, fetchLibdrmArchive, verifyLibdrmArchive, libdrmArchiveCatalog};
