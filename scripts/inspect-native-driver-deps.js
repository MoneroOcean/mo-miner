"use strict";

// Read ELF metadata only: never use ldd, load a driver, enumerate a device or execute its code.
const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {isGpuDriver} = require("./native-audit");

/** Admit normalized OS-library paths, not user/SDK directories or escaped root prefixes.
 * @param {string} file */
function systemPath(file) {
  return /^(?:\/usr)?\/lib(?:64)?(?:\/|$)/.test(file) && path.posix.normalize(file) === file &&
    !Array.from(file).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127 || char === "\\");
}

/** Parse only complete readelf dynamic-section output; malformed/missing evidence is not empty.
 * @param {string} text */
function dynamicInfo(text) {
  if (!/^Dynamic section at offset /m.test(text) || !/\(NULL\)/.test(text)) {throw new Error("Missing complete driver ELF dynamic section");}
  const values = (/** @type {string} */ tag) => [...text.matchAll(new RegExp(`\\(${tag}\\)[^\\n]*?\\[([^\\]\\n]+)\\]`, "g"))]
    .map((match) => match[1] || "");
  const needed = values("NEEDED"), sonames = values("SONAME"), search = [...values("RPATH"), ...values("RUNPATH")];
  if (sonames.length > 1 || [...needed, ...sonames].some((name) => !/^[A-Za-z0-9_+.-]+\.so(?:\.[A-Za-z0-9_+.-]+)*$/.test(name)) ||
      ["NEEDED", "SONAME", "RPATH", "RUNPATH"].some((tag) => (text.match(new RegExp(`\\(${tag}\\)`, "g")) || []).length !== values(tag).length)) {
    throw new Error("Unrecognized driver ELF dependency identity");
  }
  // Only imported versions constrain a library's OS floor; libc's exported definitions do not.
  const versionNeeds = text.includes("Version needs section") ? text.slice(text.indexOf("Version needs section")) : "";
  const glibc = [...new Set([...versionNeeds.matchAll(/\bName: GLIBC_(\S+)/g)].map((match) => match[1] || ""))];
  return {soname: sonames[0] || null, needed, search, glibc};
}

/** Honor literal origin-relative driver search paths, never evaluate shell/environment syntax.
 * @param {string[]} search @param {string} origin @param {string} [workingDirectory] */
function searchPaths(search, origin, workingDirectory) {
  return search.flatMap((value) => value.split(":")).map((value) => {
    // Empty ELF entries mean cwd, NOT the driver's directory. Only the native worker's
    // explicitly pinned root is supported; absent cwd evidence continues to fail closed.
    if (!value && workingDirectory === "/") {
      const root = fs.statSync("/");
      if (!root.isDirectory() || root.uid !== 0 || (root.mode & 0o022)) {throw new Error("Unsafe driver working directory");}
      return "/";
    }
    const expanded = value.replace(/\$\{ORIGIN\}|\$ORIGIN(?=\/|$)/g, origin);
    if (!path.posix.isAbsolute(expanded) || expanded.includes("$") || !systemPath(path.posix.normalize(expanded))) {
      throw new Error("Driver search path leaves the system library roots");
    }
    return path.posix.normalize(expanded);
  });
}

/** Inspect symlink targets too: a trusted-looking link cannot admit a writable/SDK binary.
 * @param {string} file */
function systemFile(file) {
  if (!systemPath(file)) {throw new Error("Driver dependency requires a normalized system library path: " + file);}
  const target = fs.realpathSync(file), stat = fs.statSync(target);
  if (!systemPath(target) || !stat.isFile() || stat.uid !== 0 || (stat.mode & 0o022)) {
    throw new Error("Driver dependency is not a root-owned, non-writable system file: " + file);
  }
  // Protect both the requested link's path and final binary from writable-directory swaps.
  for (const start of [path.dirname(file), path.dirname(target)]) {
    for (let directory = start;; directory = path.dirname(directory)) {
      const parent = fs.statSync(directory);
      if (!parent.isDirectory() || parent.uid !== 0 || (parent.mode & 0o022)) {throw new Error("Writable/non-system driver parent: " + directory);}
      if (directory === "/") {break;}
    }
  }
  return target;
}

/** Constant-sized hashing avoids retaining compiler-driver libraries as giant Buffers.
 * @param {string} file */
function hash(file) {
  const fd = fs.openSync(file, "r"), buffer = Buffer.allocUnsafe(1 << 20), digest = crypto.createHash("sha256");
  try {
    for (let size; (size = fs.readSync(fd, buffer, 0, buffer.length, null)) > 0;) {digest.update(buffer.subarray(0, size));}
    return digest.digest("hex");
  } finally {fs.closeSync(fd);}
}

/** Resolve only explicit ELF NEEDED edges within fixed OS roots. This is a dependency report,
 * NOT an audit exemption: late driver dlopen/JIT inputs and real runtime traces remain required.
 * @param {string[]} roots
 * @param {{lateLibraries?: string[], workingDirectory?: string}} [options] */
function inspectDrivers(roots, options = {}) {
  if (process.platform !== "linux" || !roots.length || roots.length > 12 || new Set(roots).size !== roots.length ||
      roots.some((file) => !systemPath(file) || !isGpuDriver(path.posix.basename(file)))) {
    throw new Error("Select 1..12 unique absolute system GPU-driver entrypoints");
  }
  // These late Intel GPU/compiler/sysman components are part of the installed driver,
  // not arbitrary application libraries. They never become allowed native ELF imports.
  const late = options.lateLibraries || [], intel = roots.some(file => /^lib(?:ze_intel_gpu(?:_legacy1)?\.so\.1|igdrcl(?:_legacy1)?\.so)$/.test(path.basename(file)));
  const components = new Set(["libigc.so.2", "libigdfcl.so.2", "libnl-3.so.200", "libnl-genl-3.so.200"]);
  if (!Array.isArray(late) || late.length > components.size || new Set(late).size !== late.length ||
      late.some(file => !intel || !systemPath(file) || !components.has(path.basename(file))) ||
      (options.workingDirectory !== undefined && options.workingDirectory !== "/")) {
    throw new Error("Select only explicit system Intel driver components and the verified native worker root");
  }
  const multiarch = process.arch === "x64" ? "x86_64-linux-gnu" : process.arch === "arm64" ? "aarch64-linux-gnu" : null;
  if (!multiarch) {throw new Error("Unqualified Linux driver-library architecture");}
  const defaults = [`/usr/lib/${multiarch}`, `/lib/${multiarch}`, "/usr/lib64", "/lib64", "/usr/lib", "/lib"];
  const pending = [...roots, ...late].map(systemFile), seen = new Set();
  /** @type {Array<{file: string, bytes: number, sha256: string, soname: string | null, needed: string[], search: string[], glibc: string[], dependencies: Record<string, string>}>} */
  const libraries = [];
  while (pending.length) {
    const file = /** @type {string} */ (pending.shift());
    if (seen.has(file)) {continue;}
    seen.add(file);
    if (seen.size > 128) {throw new Error("Excessive system-driver dependency graph");}
    const queried = spawnSync("readelf", ["--file-header", "--dynamic", "--version-info", "--wide", "--", file],
      {encoding: "utf8", timeout: 10000, maxBuffer: 2 << 20, env: {...process.env, LC_ALL: "C"}});
    if (queried.status !== 0) {throw new Error("Cannot inspect driver ELF: " + file, {cause: queried.error});}
    const machine = process.arch === "x64" ? /Machine:\s+Advanced Micro Devices X86-64\s*$/m : /Machine:\s+AArch64\s*$/m;
    if (!/Class:\s+ELF64\s*$/m.test(queried.stdout) || !machine.test(queried.stdout)) {throw new Error("Driver ELF architecture differs from this native host: " + file);}
    const info = dynamicInfo(queried.stdout), directories = [...searchPaths(info.search, path.dirname(file), options.workingDirectory), ...defaults];
    const dependencies = Object.fromEntries(info.needed.map((name) => {
      const candidate = directories.map((directory) => path.join(directory, name)).find((value) => fs.existsSync(value));
      if (!candidate) {throw new Error("Unresolved system-driver dependency " + name + " from " + file);}
      const target = systemFile(candidate); pending.push(target); return [name, target];
    }));
    libraries.push({file, bytes: fs.statSync(file).size, sha256: hash(file), ...info, dependencies});
  }
  return {roots: roots.map((file) => ({requested: file, resolved: systemFile(file)})),
    lateRoots: late.map(file => ({requested: file, resolved: systemFile(file)})), workingDirectory: options.workingDirectory || null,
    defaultSearch: defaults, libraries,
    scope: "Read-only system-driver ELF NEEDED and explicit late-component closure; root-owned OS files and byte hashes. Cwd is an assumption requiring separate process/trace proof. No driver/device initialization, runtime/JIT proof, or application-library exemption."};
}

module.exports = {systemPath, dynamicInfo, searchPaths, systemFile, inspectDrivers};
if (require.main === module) {
  try {console.log(JSON.stringify(inspectDrivers(process.argv.slice(2)), null, 2));} catch (error) {console.error(error); process.exitCode = 1;}
}
