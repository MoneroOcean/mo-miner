"use strict";

const path = require("node:path");
const fs = require("node:fs");
const {StringDecoder} = require("node:string_decoder");

/** Admit a regular ar signature (not thin) without loading its potentially huge payload.
 * Members, Windows import libraries and provenance remain separate build/link audits.
 * @param {string} file */
function isStaticArchive(file) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(8);
  try {
    return fs.fstatSync(fd).isFile() && fs.readSync(fd, header, 0, header.length, 0) === header.length &&
      header.toString("latin1") === "!<arch>\n";
  } finally {fs.closeSync(fd);}
}
const systemLibrary = /^(lib(c|m|dl|pthread|rt)\.so\.[0-9]+|ld-linux[^/]*\.so\.[0-9]+)$/;
// Explicit observed driver SONAMEs only: enabling CUDA never admits cudart/NVRTC/LLVM SOs.
const cudaDriverLibrary = /^(libcuda\.so\.1|libnvidia-ptxjitcompiler\.so\.1|libnvidia-ml\.so\.1)$/;
const levelZeroDriverLibrary = /^libze_intel_gpu(?:_legacy1)?\.so\.1$/;
const openclDriverLibrary = /^(libigdrcl(?:_legacy1)?\.so|libnvidia-opencl\.so\.1|libamdocl64\.so(?:\.2)?|libRusticlOpenCL\.so\.1)$/;
/** Known system GPU entrypoints, not loaders, application SDKs or arbitrary compiler libraries.
 * @param {string} name */
const isGpuDriver = (name) => cudaDriverLibrary.test(name) || levelZeroDriverLibrary.test(name) || openclDriverLibrary.test(name);
/** @typedef {{cudaDriver?: boolean, levelZeroDriver?: boolean, openclDriver?: boolean}} DriverOptions */
/** @param {string} name @param {DriverOptions} options */
const allowedLibrary = (name, options) => systemLibrary.test(name) || (options.cudaDriver === true && cudaDriverLibrary.test(name)) ||
  (options.levelZeroDriver === true && levelZeroDriverLibrary.test(name)) || (options.openclDriver === true && openclDriverLibrary.test(name));
/** Match the native ICD guard's normalized OS library paths, not arbitrary ICD environment paths.
 * Driver-transitive application/compiler dependencies still require separate provenance evidence.
 * @param {string} file */
const systemOpenclPath = (file) => /^(?:\/usr)?\/lib(?:64)?\//.test(file) &&
  !Array.from(file).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127 || char === "\\") &&
  file.split("/").slice(1).every((part) => part !== "" && part !== "." && part !== "..");
/** Keep all Linux file-open variants visible; creat can reveal forbidden extraction too.
 * @param {string} trace */
const openedFiles = (trace) => [...trace.matchAll(/\b(?:openat2?|open|creat)\([^\n]*?"([^"\n]+)"/g)].map((match) => match[1] || "");
// Loader configuration is data, not a library. Only these exact OS locations are metadata;
// a similarly named application file or an arbitrary .so probe retains the strict audit.
/** @param {string} file */
const isLibraryPath = (file) => /\.so(?:\.|$)|\.node$/.test(file) && path.basename(file) !== "ld.so.cache" &&
  !/^\/etc\/ld\.so\.conf(?:$|\.d$|\.d\/[A-Za-z0-9_.-]+\.conf$)/.test(file);

/** Inspect complete demangled nm output, not arbitrary occurrences of a nested class named node.
 * SYCL command graphs and AdaptiveCpp HCF both have legitimate ::node types. Namespace roots
 * and N-API's C ABI remain forbidden; dependency and syscall audits independently cover loads.
 * @param {string} symbols */
function auditNativeSymbols(symbols) {
  if (symbols.length > 64 * 1024 * 1024 ||
      /(?:^|[^A-Za-z0-9_:])(?:node|v8)::|\b(?:__imp_)?_?napi_[A-Za-z_]/m.test(symbols)) {
    throw new Error("Unexpected Node/V8/N-API symbols or excessive symbol data");
  }
}

/** Full static compiler graphs exceed the small in-memory audit's size limit. Scan every
 * line with bounded storage instead of truncating evidence or materializing the whole table.
 * @param {string} file @param {string[]} [requiredSymbols] */
function auditNativeSymbolFile(file, requiredSymbols = []) {
  if (!Array.isArray(requiredSymbols) || requiredSymbols.length > 64 ||
      requiredSymbols.some((name) => typeof name !== "string" || !/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name)) ||
      new Set(requiredSymbols).size !== requiredSymbols.length) {throw new Error("Invalid required native symbols");}
  const missing = new Set(requiredSymbols);
  const fd = fs.openSync(file, "r"), maximum = 512 << 20, lineLimit = 64 << 10;
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || !stat.size || stat.size > maximum) {throw new Error("Missing or excessive native symbol data");}
    const buffer = Buffer.alloc(64 << 10), decoder = new StringDecoder("utf8");
    let bytes = 0, lines = 0, tail = "";
    const inspect = (/** @type {string} */ line) => {
      if (Buffer.byteLength(line) > lineLimit) {throw new Error("Excessive native symbol line");}
      auditNativeSymbols(line); lines++;
      if (missing.size) {missing.delete(line.match(/^[0-9a-fA-F]+ [TW] ([A-Za-z_][A-Za-z0-9_]*)$/)?.[1] || "");}
    };
    for (;;) {
      const size = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (!size) {break;}
      bytes += size;
      if (bytes > maximum) {throw new Error("Excessive native symbol data");}
      const rows = (tail + decoder.write(buffer.subarray(0, size))).split("\n");
      tail = rows.pop() || "";
      for (const row of rows) {inspect(row);}
      if (Buffer.byteLength(tail) > lineLimit) {throw new Error("Excessive native symbol line");}
    }
    tail += decoder.end();
    if (tail) {inspect(tail);}
    const after = fs.fstatSync(fd);
    if (bytes !== stat.size || after.size !== stat.size || after.mtimeMs !== stat.mtimeMs || after.ctimeMs !== stat.ctimeMs) {
      throw new Error("Native symbol data changed during audit");
    }
    if (missing.size) {throw new Error("Missing linked native symbols: " + [...missing].join(", "));}
    return {bytes, lines};
  } finally {fs.closeSync(fd);}
}

/** Native feasibility lanes allow OS C/math/loader interfaces, not packaged application code.
 * This ELF check is necessary but insufficient: runtime dlopen is audited separately.
 * @param {string} section @param {DriverOptions} [options] */
function nativeDependencies(section, options = {}) {
  const needed = [...section.matchAll(/\(NEEDED\).*\[([^\]]+)\]/g)].map((match) => match[1]);
  if (/\((RPATH|RUNPATH)\)/.test(section) || needed.some((name) => !allowedLibrary(name || "", options))) {
    throw new Error(`Unexpected native application dependency/search path: ${needed.join(", ")}`);
  }
  return needed;
}

/** New references can select newer libc/libm symbol versions even when SONAMEs stay unchanged.
 * Inspect readelf --version-info against the selected build/deployment baseline, not the
 * developer host. DT_RELR is a glibc ABI feature, not a numeric symbol version: retain its
 * marker in the receipt and require the glibc release that introduced the loader support.
 * This guards accidental ABI-floor increases; it is not a clean-host run.
 * @param {string} versions @param {string} maximum */
function auditNativeGlibc(versions, maximum) {
  const parse = (/** @type {string} */ value) => {
    if (!/^\d{1,3}\.\d{1,3}(?:\.\d{1,3})?$/.test(value)) {throw new Error("Invalid native glibc baseline");}
    return value.split(".").map(Number);
  };
  const compare = (/** @type {number[]} */ actual, /** @type {number[]} */ limit) => [0, 1, 2].map((index) => (actual[index] || 0) - (limit[index] || 0))
    .find((difference) => difference !== 0) || 0;
  const limit = parse(maximum);
  if (!/No version information found|\bName: /.test(versions)) {throw new Error("Missing native ELF version evidence");}
  const required = [...versions.matchAll(/\bName: GLIBC_(\S+)/g)].map((match) => match[1] || "");
  for (const name of required) {
    if (name === "ABI_DT_RELR") {
      if (compare([2, 36], limit) > 0) {throw new Error(`Native glibc ABI GLIBC_${name} requires glibc >= 2.36; selected baseline ${maximum}`);}
      continue;
    }
    const actual = parse(name);
    const comparison = compare(actual, limit);
    if (comparison > 0) {throw new Error(`Native glibc ABI ${name} exceeds selected baseline ${maximum}`);}
  }
  return [...new Set(required)];
}

/** The executable may need symbol versions only from the allowed glibc SONAMEs, and only GLIBC_*
 * versions: a GLIBCXX_/CXXABI_/GCC_ need would mean libstdc++ or libgcc_s was linked dynamically.
 * @param {string} versions readelf --version-info output */
function auditNativeVersionNeeds(versions) {
  if (!/No version information found|\bName: /.test(versions)) {throw new Error("Missing native ELF version evidence");}
  const section = versions.match(/Version needs section '\.gnu\.version_r'[^\n]*\n([\s\S]*?)(?=\n\S|$)/)?.[1] || "";
  const files = [...section.matchAll(/\bFile: (\S+)/g)].map(match => match[1] || "");
  const names = [...section.matchAll(/\bName: (\S+)/g)].map(match => match[1] || "");
  const unexpectedFile = files.find(file => !systemLibrary.test(file)), unexpectedName = names.find(name => !/^GLIBC_/.test(name));
  if (unexpectedFile || unexpectedName) {
    throw new Error("Native executable needs a symbol version outside the allowed glibc libraries: " + (unexpectedFile || unexpectedName));
  }
  return {files: [...new Set(files)], names: [...new Set(names)]};
}

/** Reject lazy loads too, even if an optional runtime happened to be absent on the test host.
 * Callers trace openat/execve with long strings; truncated/escaped executable paths fail closed.
 * @param {string} trace @param {string} executable @param {number} [processes] @param {DriverOptions} [options] */
function auditNativeTrace(trace, executable, processes = 1, options = {}) {
  const launches = [...trace.matchAll(/\bexecve\("([^"\n]+)", \[/g)].map((match) => match[1]);
  const opened = openedFiles(trace);
  const libraries = opened.filter(isLibraryPath);
  if (!Number.isSafeInteger(processes) || processes < 1 || launches.length !== processes ||
      /\bexecveat\(/.test(trace) || (trace.match(/\bexecve\(/g) || []).length !== processes || launches.some((file) => file !== executable) ||
      opened.some((file) => file.includes("\\")) || libraries.some((file) => !allowedLibrary(path.basename(file), options) ||
        (openclDriverLibrary.test(path.basename(file)) && !systemOpenclPath(file)))) {
    throw new Error("Native syscall trace attempted an external runtime/compiler or unexpected worker launch");
  }
}

/** JIT source/library input attempts are forbidden even when unsuccessful. Exact cache roots
 * may contain generated GPU images/IR, but never source; do not mistake SDK bitcode for cache.
 * @param {string} trace @param {string[]} [includeRoots] @param {string[]} [cacheRoots] */
function auditNativeHeaders(trace, includeRoots = [], cacheRoots = []) {
  const inside = (/** @type {string} */ file, /** @type {string} */ root) => file === root || file.startsWith(root + "/");
  const roots = [...includeRoots, ...cacheRoots];
  if (roots.some((root) => !path.posix.isAbsolute(root) || root.endsWith("/") || path.posix.normalize(root) !== root)) {
    throw new Error("Header/cache audit needs exact absolute directory roots");
  }
  const opened = openedFiles(trace);
  for (const file of opened) {
    const cached = cacheRoots.some((root) => inside(file, root));
    // Original static hwloc scans readdir's "."/".." entries for NVIDIA NUMA metadata. These
    // two fixed procfs probes cannot supply compiler inputs; do not exempt any cache/SDK path.
    const numaProbe = /^\/proc\/driver\/nvidia\/gpus\/\.{1,2}\/numa_status$/.test(file);
    if (/\\/.test(file) || (!numaProbe && /(?:^|\/)\.\.?(?:\/|$)/.test(file)) || /\.(?:h|hpp|cuh|inc|c|cc|cpp|cxx|cu|hip|pch)$/i.test(file) ||
        (/\.(?:bc|hcf|ll|spv|ptx|cubin|hsaco)$/i.test(file) && !cached) ||
        /(?:^|\/)include\//.test(file) || includeRoots.some((root) => inside(file, root)) ||
        (/(?:^|\/)(?:cuda|cute|cutlass)\//.test(file) && !cached)) {
      throw new Error("Native compiler attempted a disk header or compiler resource: " + file);
    }
  }
}

module.exports = {nativeDependencies, auditNativeGlibc, auditNativeVersionNeeds, auditNativeTrace, auditNativeHeaders, auditNativeSymbols, auditNativeSymbolFile, isStaticArchive, isGpuDriver, isLibraryPath};
