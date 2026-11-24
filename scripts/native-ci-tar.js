"use strict";

// Pinned GNU tar state transport for native CI: bounded pack, list and extract of same-run work roots, and disk preflight.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");

const tar = "/usr/bin/tar", temp = "/tmp";

const linuxNativeBuildReserveBytes = 4 * 1024 * 1024 * 1024;

/** @param {string} name */
function safeRelative(name) {
  return typeof name === "string" && name.length > 0 && name.length <= 4096 && !name.startsWith("/") &&
    !name.includes("\\") && !/[\r\n\0]/.test(name) && name.split("/").every(part => part && part !== "." && part !== "..");
}

/** @param {string} command @param {string[]} args @param {number} timeout @param {number} [maxBuffer] */
function runTar(command, args, timeout = 30 * 60 * 1000, maxBuffer = 128 << 20) {
  const result = spawnSync(tar, args, {encoding: "utf8", timeout, maxBuffer});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Native CI tar " + command + " failed" + (result.stderr ? ": " + result.stderr.slice(0, 2048) : ""), {cause: result.error});
  }
  return result.stdout || "";
}

/** @param {string} base @param {string[]} paths */
function archivePaths(base, paths) {
  const resolved = fs.realpathSync(base);
  if (!Array.isArray(paths) || !paths.length || paths.length > 512 || new Set(paths).size !== paths.length ||
      paths.some(name => !safeRelative(name)) || paths.reduce((bytes, name) => bytes + Buffer.byteLength(name) + 1, 0) > (128 << 10)) {
    throw new Error("Native CI tar needs a bounded set of exact relative state paths");
  }
  if (paths.length > 8 && paths.some(name => !fs.lstatSync(path.resolve(resolved, ...name.split("/"))).isFile())) {
    throw new Error("Native CI tar permits more than eight inputs only as an explicit regular-file selection");
  }
  const roots = new Set(paths.map(name => name.split("/")[0] || ""));
  if (roots.size > 8) {throw new Error("Native CI tar input exceeds the eight-root limit");}
  for (const name of paths) {
    const item = path.resolve(resolved, ...name.split("/")), relative = path.relative(resolved, item), stat = fs.lstatSync(item);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || stat.isSymbolicLink() ||
        (!stat.isDirectory() && !stat.isFile()) || fs.realpathSync(item) !== item) {
      throw new Error("Native CI tar input escaped or is not a canonical regular state path: " + name);
    }
  }
  return resolved;
}

/** @param {string} archive @param {string} base @param {string[]} paths @param {number} [timeoutMs] @returns {{bytes: number}} */
function packState(archive, base, paths, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 30 * 60 * 1000) {throw new Error("Native CI tar timeout is outside the bounded range");}
  if (!path.isAbsolute(archive) || /[\r\n\0]/.test(archive)) {throw new Error("Native CI tar output must be an absolute path");}
  const root = archivePaths(base, paths), parent = fs.realpathSync(path.dirname(archive)), target = path.join(parent, path.basename(archive));
  // A sibling under the base is safe; only selected inputs could recursively include the output.
  if (target !== path.resolve(archive) || paths.some(name => {
    const input = path.join(root, name); return target === input || target.startsWith(input + path.sep);
  }) || fs.existsSync(target)) {
    throw new Error("Native CI tar output must be new and outside its inputs");
  }
  const fd = fs.openSync(target, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600);
  try {
    const result = spawnSync(tar, ["--create", "--file=-", "--format=posix", "--numeric-owner", "--directory", root, "--", ...paths],
      {stdio: ["ignore", fd, "inherit"], timeout: timeoutMs});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Native CI tar create failed", {cause: result.error});
    }
  } catch (error) {
    fs.closeSync(fd);
    fs.unlinkSync(target);
    throw error;
  }
  fs.closeSync(fd);
  const stat = fs.lstatSync(target);
  if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Native CI tar output is not a regular file");}
  return {bytes: stat.size};
}

/** @param {string} archive @param {string[]} expectedRoots @param {number} [timeoutMs] @returns {string[]} */
function listState(archive, expectedRoots, timeoutMs = 5 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Native CI tar listing timeout is outside the bounded range");
  }
  const listing = runTar("list", ["--list", "--file", archive, "--quoting-style=literal"], timeoutMs);
  if (listing.length > 128 << 20 || listing.includes("\0") || /(^|\n)\.[/\n]/.test(listing)) {
    throw new Error("Native CI tar member list is invalid or excessive");
  }
  const names = listing.split("\n").filter(Boolean), roots = new Set();
  if (!names.length || names.length > 500000) {throw new Error("Native CI tar member count is invalid");}
  for (const member of names) {
    const name = member.endsWith("/") ? member.slice(0, -1) : member;
    if (!safeRelative(name)) {throw new Error("Unsafe native CI tar member path");}
    roots.add(name.split("/")[0] || "");
  }
  const expected = [...new Set(expectedRoots)].sort();
  if (!expected.length || expected.some(name => !safeRelative(name) || name.includes("/")) ||
      [...roots].sort().join("\0") !== expected.join("\0")) {throw new Error("Native CI tar has unexpected state roots");}
  return names;
}

/** Extract a same-run state tar into a temporary directory, then rename selected verified roots
 * to their original absolute /tmp names. GNU tar's `--same-permissions` preserves 0444 receipts.
 * @param {string} archive @param {string} base @param {string[]} expectedRoots @param {string[]} [restoreRoots] @param {number} [timeoutMs] */
function extractState(archive, base, expectedRoots, restoreRoots = expectedRoots, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Native CI tar extraction timeout is outside the bounded range");
  }
  const deadline = Date.now() + timeoutMs;
  const root = fs.realpathSync(base), stat = fs.lstatSync(archive);
  if (!stat.isFile() || stat.isSymbolicLink() || !expectedRoots.length || expectedRoots.length > 8 ||
      new Set(expectedRoots).size !== expectedRoots.length || expectedRoots.some(name => !safeRelative(name) || name.includes("/")) ||
      !Array.isArray(restoreRoots) || !restoreRoots.length || restoreRoots.length > expectedRoots.length ||
      new Set(restoreRoots).size !== restoreRoots.length || restoreRoots.some(name => !expectedRoots.includes(name))) {
    throw new Error("Native CI tar input or expected roots are invalid");
  }
  listState(archive, expectedRoots, Math.min(5 * 60 * 1000, timeoutMs));
  for (const name of restoreRoots) {
    if (fs.existsSync(path.join(root, name))) {throw new Error("Native CI state restore refuses an existing path: " + name);}
  }
  const staging = fs.mkdtempSync(path.join(root, ".native-ci-restore-"));
  try {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {throw new Error("Native CI tar extraction deadline expired during listing");}
    runTar("extract", ["--extract", "--file", archive, "--directory", staging, "--no-same-owner", "--same-permissions", "--", ...restoreRoots], remainingMs);
    for (const name of restoreRoots) {
      const item = path.join(staging, name), itemStat = fs.lstatSync(item);
      if (!itemStat.isDirectory() || itemStat.isSymbolicLink()) {throw new Error("Native CI tar root is not a real directory: " + name);}
    }
    for (const name of restoreRoots) {fs.renameSync(path.join(staging, name), path.join(root, name));}
  } finally {
    fs.rmSync(staging, {recursive: true, force: true});
  }
}

/** Bound expanded input size before extraction. A member list also requires exact names and
 * regular files or directories in the same verbose listing used to count bytes.
 * @param {string} archive @param {number | string[]} expectedMembers @param {number} [timeoutMs] */
function tarExpandedBytes(archive, expectedMembers, timeoutMs = 15 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 15 * 60 * 1000) {
    throw new Error("Native CI tar size-listing timeout is outside the bounded range");
  }
  const names = Array.isArray(expectedMembers) ? expectedMembers : null,
    count = typeof expectedMembers === "number" ? expectedMembers : names?.length ?? 0;
  if (!Number.isSafeInteger(count) || count < 1 || count > 500000) {
    throw new Error("Native CI tar size-listing member count is outside the bounded range");
  }
  const listing = runTar("list", ["--list", "--verbose", "--numeric-owner", "--full-time", "--quoting-style=literal", "--file", archive],
      timeoutMs, names ? 8 << 20 : 128 << 20),
    lines = listing.split("\n").filter(Boolean);
  if (lines.length !== count) {
    throw new Error("Native SDK tar size listing differs from its admitted member list");
  }
  let total = 0n;
  for (const [index, line] of lines.entries()) {
    const match = /^\S+\s+\S+\s+(\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+(.*))?$/.exec(line);
    if (!match) {throw new Error("Native SDK tar size listing is malformed");}
    if (names) {
      const name = names[index] || "", directory = name.endsWith("/");
      if (!name || line[0] !== (directory ? "d" : "-") ||
          (match[2] !== name && (!directory || match[2] !== name.slice(0, -1)))) {
        throw new Error("Native CI tar contains a link, special file or ambiguous member listing");
      }
    }
    total += BigInt(match[1] || "0");
  }
  if (total > BigInt(Number.MAX_SAFE_INTEGER)) {throw new Error("Native SDK tar expanded size exceeds the safe limit");}
  return Number(total);
}

/** `expandedBytes` is the conservative sum of archive member sizes; input tars already occupy disk.
 * The app stage rechecks the free-space reserve after these roots have been restored. */
/** @param {number} expandedBytes @param {number} tempAvailableBytes @param {number} runnerTempAvailableBytes */
function linuxNativeSdkDiskPreflight(expandedBytes, tempAvailableBytes, runnerTempAvailableBytes) {
  if (![expandedBytes, tempAvailableBytes, runnerTempAvailableBytes].every(value => Number.isSafeInteger(value) && value >= 0)) {
    throw new Error("Native SDK disk preflight needs safe nonnegative byte counts");
  }
  const requiredTempBytes = expandedBytes + linuxNativeBuildReserveBytes;
  if (!Number.isSafeInteger(requiredTempBytes) || tempAvailableBytes < requiredTempBytes ||
      runnerTempAvailableBytes < linuxNativeBuildReserveBytes) {
    throw new Error("Insufficient disk for expanded native SDK inputs plus the bounded application-build reserve");
  }
  return {expandedInputBytes: expandedBytes, buildReserveBytes: linuxNativeBuildReserveBytes,
    requiredTempBytes, availableTempBytes: tempAvailableBytes, availableRunnerTempBytes: runnerTempAvailableBytes};
}

/** Read filesystem availability without rounding away the build reserve. @param {string} directory */
function availableDiskBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine native SDK filesystem block size");}
  const bytes = stat.bavail * stat.bsize;
  if (bytes > BigInt(Number.MAX_SAFE_INTEGER)) {return Number.MAX_SAFE_INTEGER;}
  return Number(bytes);
}

module.exports = {availableDiskBytes, extractState, linuxNativeBuildReserveBytes, linuxNativeSdkDiskPreflight, listState,
  packState, runTar, safeRelative, tar, tarExpandedBytes, temp};
