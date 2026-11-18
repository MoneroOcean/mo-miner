"use strict";

const fs = require("node:fs"), path = require("node:path");
const expectedMemory = Object.freeze({"memory.max": "12884901888", "memory.swap.max": "0"});
const hostReserveBytes = 536870912;

/** Parse the kernel's single MemAvailable value from /proc/meminfo.
 * @param {string} procMeminfo @returns {number} bytes
 */
function parseMemAvailableBytes(procMeminfo) {
  if (typeof procMeminfo !== "string" || procMeminfo.includes("\0") || procMeminfo.length > (1 << 20)) {
    throw new Error("Native CI host memory input is invalid");
  }
  const entries = procMeminfo.split(/\r?\n/).filter(line => line.startsWith("MemAvailable:"));
  if (entries.length !== 1) {throw new Error("Native CI host memory needs one MemAvailable entry");}
  const match = /^MemAvailable:\s+([0-9]+)\s+kB\s*$/.exec(entries[0] || ""), kibibytes = match ? Number(match[1]) : NaN,
    bytes = kibibytes * 1024;
  if (!Number.isSafeInteger(kibibytes) || kibibytes < 0 || !Number.isSafeInteger(bytes)) {
    throw new Error("Native CI host MemAvailable value is malformed or overflows");
  }
  return bytes;
}

/** Require host-wide available memory to exceed the enforced producer cap plus reserve.
 * @param {string} procMeminfo @returns {{availableBytes:number, requiredBytes:number}}
 */
function verifyNativeCiHostHeadroom(procMeminfo) {
  const availableBytes = parseMemAvailableBytes(procMeminfo), requiredBytes = Number(expectedMemory["memory.max"]) + hostReserveBytes;
  if (availableBytes <= requiredBytes) {
    throw new Error("Native CI host requires more than 12 GiB plus 512 MiB available memory");
  }
  return {availableBytes, requiredBytes};
}

/** Resolve the process's unified cgroup directory beneath the mounted cgroup-v2 root.
 * @param {string} procCgroup @param {string} mountRoot @param {typeof fs} [filesystem]
 * @returns {string} */
function verifyNativeCiMemoryEnvelope(procCgroup, mountRoot = "/sys/fs/cgroup", filesystem = fs) {
  if (typeof procCgroup !== "string" || procCgroup.includes("\0") || !path.isAbsolute(mountRoot)) {
    throw new Error("Native CI memory scope has invalid cgroup-v2 inputs");
  }
  const unified = procCgroup.split(/\r?\n/).filter(line => line.startsWith("0::"));
  const entry = unified[0];
  if (unified.length !== 1 || !entry) {throw new Error("Native CI memory scope requires one unified cgroup-v2 entry");}
  const relative = entry.slice(3), components = relative.slice(1).split("/");
  if (!relative.startsWith("/") || relative === "/" || components.some(part => !part || part === "." || part === "..")) {
    throw new Error("Native CI memory scope has an invalid unified cgroup path");
  }
  const root = filesystem.realpathSync(mountRoot), directory = path.resolve(root, ...components), resolved = filesystem.realpathSync(directory), withinRoot = path.relative(root, resolved);
  if (resolved !== directory || !withinRoot || withinRoot === ".." || withinRoot.startsWith(".." + path.sep) || path.isAbsolute(withinRoot)) {
    throw new Error("Native CI memory scope escaped the cgroup-v2 mount");
  }
  const directoryStat = filesystem.lstatSync(directory);
  if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {throw new Error("Native CI memory scope is not a cgroup directory");}
  for (const [name, expected] of Object.entries(expectedMemory)) {
    const file = path.join(directory, name), stat = filesystem.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || filesystem.readFileSync(file, "utf8").trim() !== expected) {
      throw new Error("Native CI memory scope differs from the required " + name + " envelope");
    }
  }
  return directory;
}

if (require.main === module) {
  try {
    if (process.argv[2] === "--host-headroom" && process.argv.length === 3) {
      const {availableBytes, requiredBytes} = verifyNativeCiHostHeadroom(fs.readFileSync("/proc/meminfo", "utf8"));
      process.stdout.write("[native-ci] Verified host available memory " + availableBytes + " > " + requiredBytes + " bytes\n");
    } else if (process.argv.length === 2) {
      const directory = verifyNativeCiMemoryEnvelope(fs.readFileSync("/proc/self/cgroup", "utf8"));
      process.stdout.write("[native-ci] Verified 12 GiB memory and zero swap at " + directory + "\n");
    } else {throw new Error("Invalid native CI memory-envelope mode");}
  } catch (error) {
    process.stderr.write("[native-ci] Native CI memory admission failed: " +
      (error instanceof Error ? error.message : String(error)) + "\n");
    process.exitCode = 2;
  }
}

module.exports = {parseMemAvailableBytes, verifyNativeCiHostHeadroom, verifyNativeCiMemoryEnvelope};
