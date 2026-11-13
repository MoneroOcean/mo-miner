"use strict";

// Assemble only the pinned vendor-prebuilt NVRTC inputs needed by MoM. This does not
// rebuild vendor sources or qualify redistribution rights.
const fs = require("node:fs"), path = require("node:path"), {spawn, spawnSync} = require("node:child_process");
const {Transform} = require("node:stream"), {pipeline} = require("node:stream/promises");
const {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildText, sourceInputs} = require("./build-source-inputs");
const {verifyNvrtcArchives, nvrtcArchiveCatalog, nvrtcSourceRecipe} = require("./fetch-native-sdk-sources");
const {nvrtcCohortInfo} = require("./probe-static-nvrtc");

const repo = path.resolve(__dirname, ".."), maxMemberBytes = 1536 * 1024 * 1024, maxTotalBytes = 2 * 1024 * 1024 * 1024;
const commandTimeoutMs = 10 * 60 * 1000, maxListingBytes = 8 << 20;
/** @typedef {"linux" | "windows"} Platform */
/** @typedef {"linux" | "windows" | "linux-12.6"} NvrtcCohort */
/** @typedef {"nvrtc" | "nvptxcompiler"} ArchiveName */
/** @typedef {Record<string, {sha256: string}>} ArchiveHashes */
/** @typedef {{archive: ArchiveName, member: string}} VendorMember */
/** @typedef {{output: string, source: VendorMember}} VendorPlanRow */
/** @typedef {{output: string, target: string}} AliasPlanRow */
/** @typedef {VendorPlanRow | AliasPlanRow} MemberPlanRow */
/** @typedef {{path: string, sha256: string, version: string}} ToolIdentity */
/** @typedef {{version: number, target: string, status: string, networked: boolean, buildOnly: boolean, compiled: boolean,
 * publicationQualified: boolean, fetcherSha256: string, recipe: {sha256: string, files: Record<string, string>},
 * archives: Record<string, {path: string, url: string, sha256: string, bytes: number}>}} AcquisitionReceipt */
/** @typedef {{kind: "file", bytes: number, sha256: string, source: VendorMember} |
 * {kind: "symlink", bytes: 0, target: string, source: {alias: true}}} OutputFile */
/** @typedef {Record<string, OutputFile>} OutputFiles */
/** @typedef {{version: 1, target: string, platform: Platform, vendorVersion: string, status: "PASS", networked: false,
 * buildOnly: true, compiled: false, sourceRebuilt: false, vendorPrebuilt: true, publicationQualified: false,
 * redistributionQualified: false, cohort: NvrtcCohort, producerSha256: string, recipe: {sha256: string, files: Record<string, string>}, tool: ToolIdentity,
 * source: {directory: string, receipt: AcquisitionReceipt, receiptSha256: string}, output: string, files: OutputFiles}} SdkReceipt */
/** @typedef {{receipt: SdkReceipt, receiptFile: string, receiptSha256: string, files: Record<string, string>}} SdkInputClosure */
/** @typedef {{cohort?: NvrtcCohort, hashes?: ArchiveHashes, recipe?: typeof nvrtcSdkRecipe, archiveRecipe?: typeof nvrtcSourceRecipe,
 * maxMemberBytes?: number, maxTotalBytes?: number}} SdkDependencies */

/** Fixed member paths only. Linux's existing probe expects lib64 although NVIDIA's archive
 * currently stores these files in lib/. Windows keeps the vendor lib/x64 layout. */
/** @param {Platform} platform @param {NvrtcCohort} [cohort] @returns {MemberPlanRow[]} */
function memberPlan(platform, cohort = platform) {
  if (platform !== "linux" && platform !== "windows") {throw new Error("NVRTC SDK target must be linux or windows");}
  const pins = nvrtcCohortInfo(platform, cohort);
  /** @param {ArchiveName} archive @param {string} member @param {string} output @returns {VendorPlanRow} */
  const vendor = (archive, member, output) => {
    const pin = pins.archives[archive];
    if (!pin) {throw new Error("NVRTC archive is absent from the selected cohort: " + archive);}
    const fileName = pin.fileName, suffix = platform === "linux" ? ".tar.xz" : ".zip";
    if (!fileName.endsWith(suffix)) {throw new Error("Pinned NVRTC archive name has an unexpected suffix");}
    return {output, source: {archive, member: fileName.slice(0, -suffix.length) + "/" + member}};
  };
  /** @param {string} output @param {string} target @returns {AliasPlanRow} */
  const alias = (output, target) => ({output, target});
  /** @type {MemberPlanRow[]} */ const rows = [vendor("nvrtc", "include/nvrtc.h", "include/nvrtc.h"),
    vendor("nvrtc", "LICENSE", "licenses/nvrtc/LICENSE")];
  if (cohort !== "linux-12.6") {
    rows.push(vendor("nvptxcompiler", "include/nvPTXCompiler.h", "include/nvPTXCompiler.h"),
      vendor("nvptxcompiler", "LICENSE", "licenses/nvptxcompiler/LICENSE"));
  }
  if (platform === "linux") {
    const [major, minor] = pins.version.split(".");
    rows.push(vendor("nvrtc", `lib/libnvrtc.so.${pins.version}`, `lib64/libnvrtc.so.${pins.version}`),
      vendor("nvrtc", `lib/libnvrtc-builtins.so.${pins.version}`, `lib64/libnvrtc-builtins.so.${pins.version}`),
      alias("lib64/libnvrtc.so", `libnvrtc.so.${major}`), alias(`lib64/libnvrtc.so.${major}`, `libnvrtc.so.${pins.version}`),
      alias("lib64/libnvrtc-builtins.so", `libnvrtc-builtins.so.${major}.${minor}`),
      alias(`lib64/libnvrtc-builtins.so.${major}.${minor}`, `libnvrtc-builtins.so.${pins.version}`));
    if (cohort !== "linux-12.6") {
      rows.push(vendor("nvrtc", "lib/libnvrtc_static.a", "lib64/libnvrtc_static.a"),
        vendor("nvrtc", "lib/libnvrtc-builtins_static.a", "lib64/libnvrtc-builtins_static.a"),
        vendor("nvptxcompiler", "lib/libnvptxcompiler_static.a", "lib64/libnvptxcompiler_static.a"));
    }
  } else {
    rows.push(vendor("nvrtc", "lib/x64/nvrtc_static.lib", "lib/x64/nvrtc_static.lib"),
      vendor("nvrtc", "lib/x64/nvrtc-builtins_static.lib", "lib/x64/nvrtc-builtins_static.lib"),
      vendor("nvptxcompiler", "lib/x64/nvptxcompiler_static.lib", "lib/x64/nvptxcompiler_static.lib"));
  }
  return rows.sort((a, b) => a.output.localeCompare(b.output));
}

function nvrtcSdkRecipe() {
  return sourceInputs(repo, [], ["scripts/prepare-native-nvrtc-sdk.js", "scripts/fetch-native-sdk-sources.js",
    "scripts/probe-static-nvrtc.js", "scripts/build-source-inputs.js"]);
}

/** @param {MemberPlanRow} row @returns {row is VendorPlanRow} */
function isVendorRow(row) {return "source" in row;}

/** @param {string} name */
function findTool(name) {
  for (const directory of (process.env["PATH"] || "").split(path.delimiter).filter(Boolean)) {
    try {
      const real = fs.realpathSync(path.join(directory, name));
      if (fs.statSync(real).isFile()) {fs.accessSync(real, fs.constants.X_OK); return real;}
    } catch { /* continue through PATH */ }
  }
  throw new Error("NVRTC SDK assembly requires " + name + " on PATH");
}

/** Avoid ambient tar/unzip options; arguments and selected members remain producer-owned. */
function toolEnv() {
  const env = /** @type {NodeJS.ProcessEnv} */ ({...process.env, LC_ALL: "C", LANG: "C"});
  for (const name of /** @type {const} */ (["TAR_OPTIONS", "UNZIPOPT", "UNZIP_OPTS", "ZIPINFOOPT"])) {delete env[name];}
  return env;
}

/** @param {string} tool @param {string[]} args @param {string} label @param {number} [maxBytes] */
function capture(tool, args, label, maxBytes = maxListingBytes) {
  const result = spawnSync(tool, args, {encoding: "utf8", timeout: commandTimeoutMs, maxBuffer: maxBytes, env: toolEnv()});
  if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || Buffer.byteLength(result.stdout) > maxBytes) {
    throw new Error("NVRTC SDK " + label + " failed" + (result.error ? ": " + result.error.message : ""));
  }
  return result.stdout;
}

/** @param {string} tool @param {"tar" | "unzip"} kind */
function toolIdentity(tool, kind) {
  const version = capture(tool, kind === "tar" ? ["--version"] : ["-v"], kind + " version", 64 << 10).split(/\r?\n/, 1)[0]?.trim();
  if (!version || /[\r\n\0]/.test(version)) {throw new Error("NVRTC SDK archive tool has no version identity");}
  return {path: tool, sha256: fileHash(tool), version};
}

/** Return size metadata only for each exact, pinned regular-file member. Archive paths are
 * never materialized: even unrelated or malformed names can only appear in this listing. */
/** @param {string} tool @param {string} archive @param {string[]} members @returns {Map<string, number>} */
function inspectTar(tool, archive, members) {
  const names = capture(tool, ["-tJf", archive], "tar member listing").split(/\r?\n/).filter(Boolean);
  const details = capture(tool, ["--numeric-owner", "-tvJf", archive], "tar member metadata").split(/\r?\n/).filter(Boolean);
  if (names.length !== details.length) {throw new Error("NVRTC tar listing and metadata disagree");}
  const selected = new Map();
  for (const member of members) {
    const matches = names.map((name, index) => name === member ? index : -1).filter(index => index >= 0);
    if (matches.length !== 1) {throw new Error("Pinned NVRTC tar member is missing or duplicated: " + member);}
    const index = matches[0]; if (index === undefined) {throw new Error("Pinned NVRTC tar member is missing: " + member);}
    const line = details[index] || "", match = line.match(/^-\S+\s+\S+\s+(\d+)\s/), bytes = Number(match?.[1]);
    if (!line.startsWith("-") || !Number.isSafeInteger(bytes) || bytes < 1) {
      throw new Error("Pinned NVRTC tar member is not a regular file: " + member);
    }
    selected.set(member, bytes);
  }
  return selected;
}

/** ZIP member names are matched exactly and central-directory metadata must say regular,
 * unencrypted file. unzip -p then streams only those literal names to fixed output paths. */
/** @param {string} tool @param {string} archive @param {string[]} members @returns {Map<string, number>} */
function inspectZip(tool, archive, members) {
  const names = capture(tool, ["-Z1", archive], "ZIP member listing").split(/\r?\n/).filter(Boolean);
  const details = capture(tool, ["-Z", "-v", archive], "ZIP member metadata");
  const records = details.split(/(?=^Central directory entry #[0-9]+:\s*$)/m), selected = new Map();
  for (const member of members) {
    if (names.filter(name => name === member).length !== 1) {throw new Error("Pinned NVRTC ZIP member is missing or duplicated: " + member);}
    const found = records.filter(record => record.split(/\r?\n/).some(line => line.trim() === member));
    if (found.length !== 1) {throw new Error("Pinned NVRTC ZIP member metadata is missing or duplicated: " + member);}
    const record = found[0] || "", mode = record.match(/Unix file attributes \(([0-7]+) octal\):/i)?.[1],
      size = Number(record.match(/uncompressed size:\s*([0-9]+) bytes/i)?.[1]);
    if (!/file security status:\s*not encrypted/i.test(record) || !mode ||
        (Number.parseInt(mode, 8) & 0o170000) !== 0o100000 || !Number.isSafeInteger(size) || size < 1) {
      throw new Error("Pinned NVRTC ZIP member is not an unencrypted regular file: " + member);
    }
    selected.set(member, size);
  }
  return selected;
}

/** Hash a streamed stdout member with fixed memory use and an explicit byte ceiling. */
/** @param {string} tool @param {string[]} args @param {string} destination @param {number} maximum @param {string} label
 * @returns {Promise<{bytes: number, sha256: string}>} */
function streamMember(tool, args, destination, maximum, label) {
  return new Promise((resolve, reject) => {
    const child = spawn(tool, args, {env: toolEnv(), stdio: ["ignore", "pipe", "ignore"]});
    const hash = createHash("sha256"); let bytes = 0, timedOut = false, settled = false, stopping = false;
    /** @type {NodeJS.Timeout | undefined} */ let killTimer;
    const stop = () => {if (!stopping) {stopping = true; child.kill("SIGTERM"); killTimer = setTimeout(() => child.kill("SIGKILL"), 2000);}};
    const timer = setTimeout(() => {timedOut = true; stop();}, commandTimeoutMs);
    const limiter = new Transform({transform(chunk, _encoding, callback) {
      bytes += chunk.length;
      if (bytes > maximum) {callback(new Error("NVRTC SDK member exceeds its configured byte limit: " + label)); return;}
      hash.update(chunk); callback(null, chunk);
    }});
    const output = fs.createWriteStream(destination, {flags: "wx", mode: 0o600}), writing = pipeline(child.stdout, limiter, output);
    writing.catch(stop);
    const closed = new Promise((resolveClose, rejectClose) => {
      child.once("error", rejectClose); child.once("close", (code, signal) => resolveClose({code, signal}));
    });
    Promise.allSettled([writing, closed]).then(results => {
      if (settled) {return;} settled = true; clearTimeout(timer); clearTimeout(killTimer);
      const written = results[0], exited = results[1];
      if (!written || written.status === "rejected") {reject(written?.status === "rejected" ? written.reason : new Error("NVRTC SDK write did not settle: " + label)); return;}
      if (!exited || exited.status === "rejected") {reject(exited?.status === "rejected" ? exited.reason : new Error("NVRTC SDK child did not settle: " + label)); return;}
      const exit = /** @type {{code: number | null, signal: NodeJS.Signals | null}} */ (exited.value);
      if (timedOut || exit.code !== 0) {reject(new Error("NVRTC SDK stream failed" + (timedOut ? " (deadline exceeded)" : ` (${exit.code ?? exit.signal})`) + ": " + label)); return;}
      resolve({bytes, sha256: hash.digest("hex")});
    }, error => {
      if (settled) {return;} settled = true; clearTimeout(timer); clearTimeout(killTimer);
      reject(error instanceof Error ? error : new Error("NVRTC SDK stream failed: " + label));
    });
  });
}

/** @param {string} output */
function newDestination(output) {
  if (typeof output !== "string" || !path.isAbsolute(output) || /[\r\n\0]/.test(output)) {throw new Error("NVRTC SDK output needs a new absolute directory outside the repository");}
  const parent = fs.realpathSync(path.dirname(output)), destination = path.join(parent, path.basename(output));
  let exists = false; try {fs.lstatSync(destination); exists = true;} catch (error) {if (Reflect.get(Object(error), "code") !== "ENOENT") {throw error;}}
  if (destination !== path.resolve(output) || exists || destination === repo || destination.startsWith(repo + path.sep)) {
    throw new Error("NVRTC SDK output must be new, canonical, and outside the repository");
  }
  return destination;
}

/** @param {string} output */
function existingDestination(output) {
  if (typeof output !== "string" || !path.isAbsolute(output) || /[\r\n\0]/.test(output)) {throw new Error("NVRTC SDK admission needs an absolute path");}
  const directory = path.resolve(output), stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory || directory === repo || directory.startsWith(repo + path.sep)) {
    throw new Error("NVRTC SDK output must be a canonical real directory outside the repository");
  }
  return directory;
}

/** Keep existing source hashes bounded and refuse linked or writable SDK files.
 * @param {string} file @param {number} maximum @returns {{bytes: number, sha256: string}} */
function readonlyHash(file, maximum) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n || (before.mode & 0o222n) !== 0n ||
      before.size < 1n || before.size > BigInt(maximum)) {throw new Error("NVRTC SDK file is not bounded, read-only, and single-link: " + file);}
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size || before.mode !== after.mode ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || after.nlink !== 1n) {
    throw new Error("NVRTC SDK file changed while hashing: " + file);
  }
  return {bytes: Number(after.size), sha256};
}

/** In-memory identities catch replacements/restores during one assembly or admission pass;
 * they are deliberately not persisted, so copied SDKs are admitted by content and paths. */
/** @param {string[]} paths @returns {Array<{file: string, stat: import("node:fs").BigIntStats}>} */
function snapshotPaths(paths) {
  return [...new Set(paths)].map(file => {
    const stat = fs.lstatSync(file, {bigint: true});
    if (!stat.isFile() && !stat.isDirectory() && !stat.isSymbolicLink()) {throw new Error("Unexpected NVRTC SDK filesystem object: " + file);}
    return {file, stat};
  });
}

/** @param {ReturnType<typeof snapshotPaths>} snapshots @param {string} label */
function assertSnapshots(snapshots, label) {
  for (const {file, stat} of snapshots) {
    const after = fs.lstatSync(file, {bigint: true});
    if (stat.dev !== after.dev || stat.ino !== after.ino || stat.mode !== after.mode || stat.size !== after.size ||
        stat.nlink !== after.nlink || stat.mtimeNs !== after.mtimeNs || stat.ctimeNs !== after.ctimeNs) {
      throw new Error("NVRTC SDK " + label + " metadata changed during admission: " + file);
    }
  }
}

/** @param {string} file @returns {{value: unknown, sha256: string}} */
function readReceipt(file) {
  const digest = readonlyHash(file, 1 << 20), value = JSON.parse(readBuildText(file, 1 << 20));
  if (fileHash(file) !== digest.sha256) {throw new Error("NVRTC SDK receipt changed while reading");}
  return {value, sha256: digest.sha256};
}

/** @param {unknown} value @param {string[]} names */
function exactKeys(value, names) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) &&
    isDeepStrictEqual(Object.keys(value).sort(), [...names].sort()));
}

/** @param {unknown} value @returns {value is SdkReceipt} */
function isSdkReceipt(value) {
  if (!exactKeys(value, ["version", "target", "platform", "cohort", "vendorVersion", "status", "networked", "buildOnly", "compiled", "sourceRebuilt",
    "vendorPrebuilt", "publicationQualified", "redistributionQualified", "producerSha256", "recipe", "tool", "source", "output", "files"])) {return false;}
  const row = /** @type {Record<string, unknown>} */ (value);
  return row["version"] === 1 && (row["platform"] === "linux" || row["platform"] === "windows") &&
    (row["cohort"] === "linux" || row["cohort"] === "windows" || row["cohort"] === "linux-12.6") && typeof row["target"] === "string" &&
    typeof row["vendorVersion"] === "string" && row["status"] === "PASS" && typeof row["output"] === "string" &&
    typeof row["producerSha256"] === "string" && typeof row["files"] === "object" && row["files"] !== null;
}

/** @param {Platform} platform @param {string} directory @param {ArchiveHashes | undefined} hashes
 * @param {typeof nvrtcSourceRecipe} recipe @param {NvrtcCohort} cohort @returns {ReturnType<typeof verifyNvrtcArchives>} */
function verifyAcquisition(platform, directory, hashes, recipe, cohort) {
  return verifyNvrtcArchives(platform, directory, {...(hashes === undefined ? {} : {hashes}), recipe, cohort});
}

/** @param {Platform} platform @param {string} sourceDirectory @param {string} output
 * @param {SdkDependencies} [dependencies] @returns {Promise<SdkReceipt>} */
async function prepareNvrtcSdk(platform, sourceDirectory, output, dependencies = {}) {
  if (process.platform !== "linux") {throw new Error("NVRTC SDK assembly runs on Linux for either SDK target");}
  const cohort = dependencies.cohort === undefined ? platform : dependencies.cohort, pin = nvrtcCohortInfo(platform, cohort),
    plan = memberPlan(platform, cohort), destination = newDestination(output);
  if (typeof sourceDirectory !== "string" || !path.isAbsolute(sourceDirectory) || path.resolve(sourceDirectory) !== sourceDirectory ||
      /[\r\n\0]/.test(sourceDirectory) || destination === sourceDirectory || destination.startsWith(sourceDirectory + path.sep)) {
    throw new Error("NVRTC SDK assembly needs a canonical verified source directory separate from output");
  }
  const memberLimit = dependencies.maxMemberBytes ?? maxMemberBytes, totalLimit = dependencies.maxTotalBytes ?? maxTotalBytes;
  if (!Number.isSafeInteger(memberLimit) || memberLimit < 1 || memberLimit > maxMemberBytes ||
      !Number.isSafeInteger(totalLimit) || totalLimit < 1 || totalLimit > maxTotalBytes) {throw new Error("NVRTC SDK limits may only be narrowed");}
  const archiveRecipe = dependencies.archiveRecipe || nvrtcSourceRecipe;
  const acquisition = verifyAcquisition(platform, sourceDirectory, dependencies.hashes, archiveRecipe, cohort);
  const sourceReceiptPath = path.join(sourceDirectory, "fetch.json"), sourceReceiptSha256 = fileHash(sourceReceiptPath);
  const source = {directory: sourceDirectory, receipt: acquisition, receiptSha256: sourceReceiptSha256};
  const recipeFor = dependencies.recipe || nvrtcSdkRecipe, recipe = recipeFor(), recipeJson = JSON.stringify(recipe), producerSha256 = fileHash(__filename);
  const kind = platform === "linux" ? "tar" : "unzip", tool = findTool(kind), toolInfo = toolIdentity(tool, kind),
    catalog = nvrtcArchiveCatalog(platform, dependencies.hashes, cohort);
  const inputSnapshots = snapshotPaths([sourceDirectory, sourceReceiptPath, ...Object.values(acquisition.archives).map(row => row.path), tool]);
  /** @type {Record<string, Map<string, number>>} */ const metadata = {};
  for (const name of Object.keys(catalog).sort()) {
    const archive = acquisition.archives[name], members = plan.filter(isVendorRow).filter(row => row.source.archive === name).map(row => row.source.member);
    if (!archive || members.some(member => !member)) {throw new Error("NVRTC SDK source receipt has an incomplete pinned cohort");}
    metadata[name] = kind === "tar" ? inspectTar(tool, archive.path, members) : inspectZip(tool, archive.path, members);
  }
  fs.mkdirSync(destination, {mode: 0o700});
  let total = 0;
  /** @type {OutputFiles} */ const files = {};
  for (const row of plan) {
    const target = path.join(destination, row.output); fs.mkdirSync(path.dirname(target), {recursive: true, mode: 0o700});
    if ("target" in row) {
      // These relative SONAME aliases are fixed by the pinned version; do not trust or extract vendor symlink entries.
      fs.symlinkSync(row.target, target);
      files[row.output] = {kind: "symlink", bytes: 0, target: row.target, source: {alias: true}};
      continue;
    }
    const sourceMember = row.source;
    const archive = acquisition.archives[sourceMember.archive], memberSize = metadata[sourceMember.archive]?.get(sourceMember.member);
    if (!archive || memberSize === undefined) {throw new Error("Pinned NVRTC member identity is missing: " + sourceMember.member);}
    if (memberSize > memberLimit || total + memberSize > totalLimit) {throw new Error("NVRTC SDK selected files exceed their extraction limit");}
    const args = kind === "tar" ? ["-xJOf", archive.path, "--", sourceMember.member] : ["-p", archive.path, sourceMember.member];
    const result = /** @type {{bytes: number, sha256: string}} */ (await streamMember(tool, args, target, memberLimit, sourceMember.member));
    if (result.bytes !== memberSize) {throw new Error("NVRTC SDK member size differs from its original metadata: " + sourceMember.member);}
    fs.chmodSync(target, 0o444); total += result.bytes;
    files[row.output] = {kind: "file", bytes: result.bytes, sha256: result.sha256,
      source: {archive: sourceMember.archive, member: sourceMember.member}};
  }
  if (!isDeepStrictEqual(recipeFor(), recipe) || fileHash(__filename) !== producerSha256 || fileHash(tool) !== toolInfo.sha256) {
    throw new Error("NVRTC SDK producer, tool, or recipe changed during assembly");
  }
  const acquisitionAfter = verifyAcquisition(platform, sourceDirectory, dependencies.hashes, archiveRecipe, cohort);
  if (!isDeepStrictEqual(acquisitionAfter, acquisition) || fileHash(sourceReceiptPath) !== sourceReceiptSha256) {
    throw new Error("NVRTC SDK original acquisition changed during assembly");
  }
  assertSnapshots(inputSnapshots, "input");
  const directories = [...new Set(plan.flatMap(row => {
    const parts = row.output.split("/"); return parts.slice(0, -1).map((_part, index) => parts.slice(0, index + 1).join("/"));
  }))].sort((a, b) => b.length - a.length);
  for (const relative of directories) {fs.chmodSync(path.join(destination, relative), 0o555);}
  const receipt = /** @type {SdkReceipt} */ ({version: 1, target: "nvrtc-sdk-" + cohort, platform, cohort, vendorVersion: pin.version,
    status: "PASS", networked: false, buildOnly: true, compiled: false, sourceRebuilt: false, vendorPrebuilt: true,
    publicationQualified: false, redistributionQualified: false, producerSha256,
    recipe: {sha256: createHash("sha256").update(recipeJson).digest("hex"), files: recipe.graph},
    tool: toolInfo, source, output: destination, files});
  fs.writeFileSync(path.join(destination, "sdk.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.chmodSync(path.join(destination, "sdk.json"), 0o444); fs.chmodSync(destination, 0o555);
  return receipt;
}

/** Validate the exact read-only output tree and continuity back to the unchanged acquisition.
 * @param {Platform} platform @param {string} output @param {SdkDependencies} [dependencies] @returns {SdkReceipt} */
function admitNvrtcSdk(platform, output, dependencies = {}) {
  const cohort = dependencies.cohort === undefined ? platform : dependencies.cohort, pin = nvrtcCohortInfo(platform, cohort), plan = memberPlan(platform, cohort),
    directory = existingDestination(output), outputPaths = [directory, path.join(directory, "sdk.json")];
  for (const row of plan) {
    outputPaths.push(path.join(directory, row.output));
    let parent = path.dirname(row.output);
    while (parent !== ".") {outputPaths.push(path.join(directory, parent)); parent = path.dirname(parent);}
  }
  const outputSnapshots = snapshotPaths(outputPaths), saved = readReceipt(path.join(directory, "sdk.json"));
  if (!isSdkReceipt(saved.value)) {throw new Error("NVRTC SDK receipt has an invalid schema");}
  const receipt = saved.value, recipeFor = dependencies.recipe || nvrtcSdkRecipe, recipe = recipeFor(),
    recipeHash = createHash("sha256").update(JSON.stringify(recipe)).digest("hex");
  if (receipt.version !== 1 || receipt.target !== "nvrtc-sdk-" + cohort || receipt.platform !== platform || receipt.cohort !== cohort ||
      receipt.vendorVersion !== pin.version || receipt.status !== "PASS" || receipt.networked !== false ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.sourceRebuilt !== false || receipt.vendorPrebuilt !== true ||
      receipt.publicationQualified !== false || receipt.redistributionQualified !== false || receipt.output !== directory) {
    throw new Error("NVRTC SDK receipt is not a valid offline vendor-prebuilt assembly");
  }
  if (!exactKeys(receipt.recipe, ["sha256", "files"]) || receipt.producerSha256 !== fileHash(__filename) || receipt.recipe?.sha256 !== recipeHash ||
      !isDeepStrictEqual(receipt.recipe.files, recipe.graph)) {throw new Error("NVRTC SDK receipt has stale producer recipe provenance");}
  const toolKind = platform === "linux" ? "tar" : "unzip", actualTool = toolIdentity(findTool(toolKind), toolKind);
  const toolSnapshots = snapshotPaths([actualTool.path]);
  if (!isDeepStrictEqual(receipt.tool, actualTool)) {throw new Error("NVRTC SDK receipt has stale archive-tool identity");}
  const sourceDirectory = receipt.source?.directory, sourceReceipt = receipt.source?.receipt;
  if (!exactKeys(receipt.source, ["directory", "receipt", "receiptSha256"]) || typeof sourceDirectory !== "string" ||
      !/^[0-9a-f]{64}$/.test(receipt.source.receiptSha256 || "") || !path.isAbsolute(sourceDirectory) || path.resolve(sourceDirectory) !== sourceDirectory ||
      directory === sourceDirectory || directory.startsWith(sourceDirectory + path.sep)) {throw new Error("NVRTC SDK receipt has an invalid source directory");}
  const sourceReceiptPath = path.join(sourceDirectory, "fetch.json"), sourceArchives = Object.values(nvrtcArchiveCatalog(platform, dependencies.hashes, cohort))
      .map(row => path.join(sourceDirectory, row.fileName)), sourceSnapshots = snapshotPaths([sourceDirectory, sourceReceiptPath, ...sourceArchives]);
  const acquisition = verifyAcquisition(platform, sourceDirectory, dependencies.hashes, dependencies.archiveRecipe || nvrtcSourceRecipe, cohort);
  if (!isDeepStrictEqual(acquisition, sourceReceipt) || fileHash(path.join(sourceDirectory, "fetch.json")) !== receipt.source.receiptSha256) {
    throw new Error("NVRTC SDK receipt no longer matches its original acquisition");
  }
  const expectedFiles = plan.map(row => row.output).sort(), /** @type {string[]} */ actualFiles = [], expectedDirs = new Set([""]);
  for (const row of plan) {const parts = row.output.split("/"); for (let i = 1; i < parts.length; i++) {expectedDirs.add(parts.slice(0, i).join("/"));}}
  const expectedNames = new Set([...expectedFiles, "sdk.json"]);
  /** @param {string} parent @param {string} relative */
  function visit(parent, relative) {
    const stat = fs.lstatSync(parent);
    if (!stat.isDirectory() || stat.isSymbolicLink() || (stat.mode & 0o222) !== 0) {throw new Error("NVRTC SDK directory is not read-only: " + parent);}
    for (const name of fs.readdirSync(parent)) {
      const childRelative = relative ? relative + "/" + name : name, child = path.join(parent, name), childStat = fs.lstatSync(child);
      if (childStat.isDirectory() && !childStat.isSymbolicLink()) {
        if (!expectedDirs.has(childRelative)) {throw new Error("NVRTC SDK has an unexpected directory: " + childRelative);}
        visit(child, childRelative);
      } else {
        if (!expectedNames.has(childRelative)) {throw new Error("NVRTC SDK has an unexpected output: " + childRelative);}
        actualFiles.push(childRelative);
      }
    }
  }
  visit(directory, ""); actualFiles.sort();
  if (!isDeepStrictEqual(actualFiles, [...expectedFiles, "sdk.json"].sort())) {throw new Error("NVRTC SDK output tree is incomplete");}
  for (const row of plan) {
    const savedFile = receipt.files?.[row.output], file = path.join(directory, row.output);
    if ("target" in row) {
      if (!savedFile || savedFile.kind !== "symlink" || !exactKeys(savedFile, ["kind", "bytes", "target", "source"]) || savedFile.bytes !== 0 || savedFile.target !== row.target ||
          !isDeepStrictEqual(savedFile.source, {alias: true})) {throw new Error("NVRTC SDK alias receipt is invalid: " + row.output);}
      const stat = fs.lstatSync(file);
      if (!stat.isSymbolicLink() || stat.nlink !== 1 || fs.readlinkSync(file) !== row.target) {throw new Error("NVRTC SDK SONAME alias differs: " + row.output);}
    } else {
      const source = row.source;
      if (!savedFile || savedFile.kind !== "file" || !exactKeys(savedFile, ["kind", "bytes", "sha256", "source"]) || !isDeepStrictEqual(savedFile.source, source)) {
        throw new Error("NVRTC SDK member receipt is invalid: " + row.output);
      }
      const actual = readonlyHash(file, maxMemberBytes);
      if (actual.bytes !== savedFile.bytes || actual.sha256 !== savedFile.sha256) {throw new Error("NVRTC SDK output hash differs from receipt: " + row.output);}
    }
  }
  const expectedManifest = Object.fromEntries(plan.map(row => [row.output, receipt.files[row.output]]));
  if (!isDeepStrictEqual(receipt.files, expectedManifest)) {throw new Error("NVRTC SDK receipt has extra or missing output members");}
  assertSnapshots([...outputSnapshots, ...sourceSnapshots, ...toolSnapshots], "receipt/source/output/tool");
  return receipt;
}

/** Flatten an admitted SDK and its original provenance into the bounded file-hash closure
 * consumers can recheck after linking. Linux SONAME aliases are the only omitted outputs.
 * @param {Platform} platform @param {string} output @param {SdkDependencies} [dependencies] @returns {SdkInputClosure} */
function nvrtcSdkInputs(platform, output, dependencies = {}) {
  const cohort = dependencies.cohort === undefined ? platform : dependencies.cohort;
  const directory = existingDestination(output);
  if (directory !== output) {throw new Error("NVRTC SDK input root must be canonical");}
  const receiptFile = path.join(directory, "sdk.json"), receiptSha256 = fileHash(receiptFile),
    receipt = admitNvrtcSdk(platform, directory, {...dependencies, cohort});
  if (receipt.output !== directory || fileHash(receiptFile) !== receiptSha256) {throw new Error("NVRTC SDK receipt changed during input admission");}
  /** @type {Record<string, string>} */ const files = {[receiptFile]: receiptSha256};
  for (const row of memberPlan(platform, cohort)) {
    if (!isVendorRow(row)) {continue;}
    const member = receipt.files[row.output];
    if (!member || member.kind !== "file" || !/^[a-f0-9]{64}$/.test(member.sha256)) {throw new Error("NVRTC SDK output is not a recorded regular file: " + row.output);}
    files[path.join(directory, row.output)] = member.sha256;
  }
  const sourceReceiptFile = path.join(receipt.source.directory, "fetch.json");
  files[sourceReceiptFile] = receipt.source.receiptSha256;
  for (const archive of Object.values(receipt.source.receipt.archives)) {files[archive.path] = archive.sha256;}
  files[receipt.tool.path] = receipt.tool.sha256;
  // The admission routine compares this receipt cohort to the canonical four repo-relative inputs.
  for (const [relative, hash] of Object.entries(receipt.recipe.files)) {files[path.resolve(repo, ...relative.split("/"))] = hash;}
  for (const [file, hash] of Object.entries(files)) {
    if (!/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("NVRTC SDK input changed during final admission: " + file);}
  }
  return {receipt, receiptFile, receiptSha256, files};
}

if (require.main === module) {
  (async () => {
    try {
      const [target, first, second, extra] = process.argv.slice(2), prepare = target === "linux" || target === "windows" || target === "linux-12.6",
        verify = target === "verify-linux" || target === "verify-windows" || target === "verify-linux-12.6",
        platform = /** @type {Platform} */ (target === "windows" || target === "verify-windows" ? "windows" : "linux"),
        cohort = /** @type {NvrtcCohort} */ (target?.endsWith("linux-12.6") ? "linux-12.6" : platform);
      if (prepare && first && second && !extra) {
        const receipt = await prepareNvrtcSdk(platform, first, second, {cohort});
        console.log(`[native-sdk] Assembled offline vendor-prebuilt NVRTC ${cohort} SDK at ${receipt.output}; no rebuild or redistribution qualification.`);
      } else if (verify && first && !second && !extra) {
        admitNvrtcSdk(platform, first, {cohort});
        console.log(`[native-sdk] Verified offline vendor-prebuilt NVRTC ${cohort} SDK and output receipt; no rebuild or redistribution qualification.`);
      } else {
        throw new Error("Usage: prepare-native-nvrtc-sdk.js linux|linux-12.6|windows /verified/vendor-input /new/sdk-output | verify-linux|verify-linux-12.6|verify-windows /sdk-output");
      }
    } catch (error) {console.error(error); process.exitCode = 1;}
  })();
}

module.exports = {prepareNvrtcSdk, admitNvrtcSdk, nvrtcSdkInputs, nvrtcSdkRecipe, memberPlan, streamMember};
