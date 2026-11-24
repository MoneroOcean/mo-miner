"use strict";

const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");

const includeOptions = ["-idirafter", "-isystem", "-iquote", "-imsvc", "-I"];

/** Extract explicit include directories and CUDA's implied include directory from compiler argv.
 * @param {string[]} args */
function compilerIncludePaths(args) {
  const paths = [];
  for (let index = 0; index < args.length; ++index) {
    const arg = args[index] || "";
    if (arg.startsWith("--cuda-path=")) {
      paths.push(path.join(arg.slice("--cuda-path=".length), "include"));
      continue;
    }
    const option = includeOptions.find(flag => arg.startsWith(flag));
    if (option) {
      const selected = arg === option ? args[++index] : arg.slice(option.length);
      if (selected) {paths.push(selected);}
    }
  }
  return paths;
}

/** Check lexical containment, including the root itself. Callers resolve symlinks when needed.
 * @param {string} root @param {string} file */
function containsBuildPath(root, file) {
  const relative = path.relative(root, file);
  return relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}

/** @param {string} left @param {string} right */
function buildPathsOverlap(left, right) {
  return containsBuildPath(left, right) || containsBuildPath(right, left);
}

/** Optional include flags may point to missing paths; other filesystem errors block caching.
 * @param {string} directory */
function isBuildIncludeDirectory(directory) {
  try {return fs.statSync(directory).isDirectory();} catch (error) {
    if (!(error instanceof Error) || !("code" in error) || !["ENOENT", "ENOTDIR"].includes(String(error.code))) {throw error;}
    return false;
  }
}

/** Keep the default and the 1..16 bound aligned across native SDK producer entry points.
 * @param {NodeJS.ProcessEnv} [environment] */
function nativeBuildJobs(environment = process.env) {
  const value = environment["MOM_BUILD_JOBS"] || "2";
  if (!/^(?:[1-9]|1[0-6])$/.test(value)) {
    throw new Error("Native builds need MOM_BUILD_JOBS as an integer in 1..16");
  }
  return Number(value);
}

/** SDK archives can be gigabytes. Hash regular files with bounded storage and reject a
 * changing input rather than publishing a digest of an inconsistent build dependency.
 * @param {string} file */
function fileHash(file) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile()) {throw new Error("Build input is not a regular file: " + file);}
    const buffer = Buffer.allocUnsafe(64 << 10), hash = createHash("sha256");
    let bytes = 0n;
    while (bytes <= before.size) {
      const remaining = before.size - bytes + 1n;
      const size = fs.readSync(fd, buffer, 0, Number(remaining < BigInt(buffer.length) ? remaining : BigInt(buffer.length)), null);
      if (!size) {break;}
      bytes += BigInt(size); hash.update(buffer.subarray(0, size));
    }
    const after = fs.fstatSync(fd, {bigint: true});
    if (bytes !== before.size || after.size !== before.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) {
      throw new Error("Build input changed while hashing: " + file);
    }
    return hash.digest("hex");
  } finally {fs.closeSync(fd);}
}

/** Keep directory names in code-unit order for build receipts and cache keys.
 * @param {string} directory */
function sortedBuildEntries(directory) {
  return fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) =>
    a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
}

/** Shared source/SDK traversal. Callers choose the source roots explicitly.
 * @param {string} directory @param {(file: string) => void} visit */
function walk(directory, visit) {
  for (const entry of sortedBuildEntries(directory)) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {walk(file, visit);} else {visit(file);}
  }
}

const engineeringGraph = ["binding.gyp", "scripts/linux-build-state.js", "scripts/build-source-inputs.js",
  "scripts/build-helpers.sh", "scripts/build-environment.js", "scripts/cpu-cflags.sh", "scripts/cpu-optflags.sh",
  "scripts/cpu-feature.sh", "scripts/cxx-combined.sh", "scripts/cxx-adaptivecpp.sh", "scripts/cxx-oneapi.sh",
  "scripts/combined-build.sh", "scripts/adaptivecpp-entrypoint.sh", "scripts/multicompiler-entrypoint.sh"];

/** Source membership and contents, not mtimes: added/removed headers and backdated edits
 * invalidate a receipt too. Application JS is deliberately absent from Engineering inputs.
 * @param {string} root @param {string[]} [directories] @param {string[]} [leaves] */
function sourceInputs(root, directories = ["native", "sycl", "xmrig"], leaves = engineeringGraph) {
  if (!path.isAbsolute(root) || [...directories, ...leaves].some(name => typeof name !== "string" ||
      /[\\:\0]/.test(name) || name.split("/").some(part => !part || part === "." || part === ".."))) {
    throw new Error("Build sources need an absolute root and explicit relative input names");
  }
  /** @type {Record<string, string>} */
  const files = {};
  for (const directory of directories) {
    const source = path.join(root, directory);
    if (!fs.lstatSync(source).isDirectory()) {throw new Error("Build source root is not a real directory: " + source);}
    walk(source, file => {files[path.relative(root, file).split(path.sep).join("/")] = fileHash(file);});
  }
  const graph = Object.fromEntries(leaves.map(name => [name, fileHash(path.join(root, name))]));
  for (const [name, value] of Object.entries(files)) {if (/\.gypi?$/.test(name)) {graph[name] = value;}}
  return {files, graph};
}

/** Native kernel/Core caches share content tracking, but exclude the fresh AOT controller.
 * Include the whole selected source subtree so newly introduced conditional headers count.
 * @param {string} root @param {"cpu" | "gpu"} kind */
function nativeSourceInputs(root, kind) {
  if (!["cpu", "gpu"].includes(kind)) {throw new Error("Unknown native source scope");}
  const common = ["binding.gyp", "package.json", "scripts/build-source-inputs.js", "scripts/native-object-cache.js"];
  const leaves = kind === "cpu" ? ["scripts/build-native-core.js", "scripts/build-native-windows-core.js",
    "scripts/windows-incremental-build.ps1", "scripts/windows-msbuild-cache.ps1",
    "scripts/native-gpu-table.js", "scripts/native-compiler-isolation.js", "scripts/native-audit.js", "sycl/lib.h",
    "scripts/cpu-cflags.sh", "scripts/cpu-optflags.sh", "scripts/cpu-feature.sh"] :
    ["scripts/native-gpu-build.js", "scripts/build-native-gpu.js", "scripts/build-native-windows-gpu.js", "scripts/native-windows-wrapper.js",
      "scripts/build-native-windows-controller.js", "scripts/build-native-release.js", "scripts/build-environment.js",
      "runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch",
      "tests/native/dpcpp-static/static-windows.h", "scripts/native-gpu-table.js",
      "scripts/native-compiler-isolation.js", "scripts/native-audit.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js",
      // Payload hashing and static-JIT admission are part of GPU production even though
      // imported by helpers; changing either must invalidate a previously audited object.
      "scripts/native-elf.js", "scripts/native-jit-cache-tag.js", "scripts/native-jit-link.js", "scripts/build-native-windows-hiprtc-sdk.js",
      "scripts/native-dpcpp-pins.js", "scripts/native-comgr-inputs.js", "scripts/native-resources.js", "scripts/native-jit-headers.js",
      "scripts/native-cuda-images.js", "scripts/native-cuda-images.cpp", "scripts/native-linux-sysroot.js",
      "scripts/check-portable-opencl.js", "scripts/cxx-combined.sh", "scripts/combined-build.sh", "runtime/driver-library.cpp",
      "runtime/compiler-gate.cpp", "runtime/system-driver.h", "runtime/static-constructors.ld"];
  // Kernels also include shared native headers (constants, job bounds, JIT headers and CPU
  // scheduling). Inventory that whole tree, including future conditional headers, in both lanes.
  return sourceInputs(root, kind === "cpu" ? ["native", "xmrig"] : ["sycl", "native", "xmrig"], [...common, ...leaves]);
}

/** A changed or legacy receipt must request a rebuild, never bless cached objects with a new
 * timestamp/hash. Keep diagnostics bounded; formatting large source diffs can itself OOM.
 * @param {unknown} expected @param {ReturnType<typeof sourceInputs>} actual @param {string} label */
function verifySourceInputs(expected, actual, label) {
  if (!isDeepStrictEqual(expected, actual)) {throw new Error(label + " source inputs are stale or missing; rebuild this component");}
}

/** @param {import("node:fs").Stats} a @param {import("node:fs").Stats} b */
function sameBuildFile(a, b) {
  return a.dev === b.dev && a.ino === b.ino && a.size === b.size && a.nlink === b.nlink &&
    a.mtimeMs === b.mtimeMs && a.ctimeMs === b.ctimeMs;
}

/** Freeze one fd and reject growth or truncation. The 256 MiB ceiling covers the largest
 * compiler logs and compile databases. Non-regular files cannot block a build preflight.
 * @param {string} file @param {number} [maximum] @param {import("node:fs").Stats} [expected] */
function readBuildBytes(file, maximum = 8 << 20, expected) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const stat = fs.fstatSync(fd);
    if (expected && !sameBuildFile(stat, expected)) {throw new Error("Build receipt changed before admission: " + file);}
    if (!Number.isSafeInteger(maximum) || maximum < 1 || maximum > (256 << 20) || !stat.isFile() || stat.size < 1 || stat.size > maximum) {
      throw new Error("Invalid or excessive build receipt: " + file);
    }
    const data = Buffer.alloc(stat.size + 1);
    let size = 0;
    for (let n; size < data.length && (n = fs.readSync(fd, data, size, data.length - size, null));) {size += n;}
    const after = fs.fstatSync(fd);
    if (size !== stat.size || !sameBuildFile(stat, after)) {
      throw new Error("Build receipt changed during admission: " + file);
    }
    return data.subarray(0, size);
  } finally {fs.closeSync(fd);}
}

/** @param {string} file @param {number} [maximum] */
function readBuildText(file, maximum = 8 << 20) {return readBuildBytes(file, maximum).toString("utf8");}

/** @param {string} file @param {number} [maximum] */
function readBuildJson(file, maximum) {return JSON.parse(readBuildText(file, maximum));}

/** Retain the digest of the bytes actually parsed so callers can compare it with a receipt pin.
 * @param {string} file @param {number} [maximum] @param {{sha256:string, message:string}} [pin] */
function readBuildJsonSnapshot(file, maximum, pin) {
  const bytes = readBuildBytes(file, maximum), sha256 = createHash("sha256").update(bytes).digest("hex");
  if (pin && sha256 !== pin.sha256) {throw new Error(pin.message);}
  return {value: JSON.parse(bytes.toString("utf8")), sha256};
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} label */
function requireBuildFields(value, keys, label) {
  if (!hasExactBuildFields(value, keys)) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}

/** Compare field names directly because JSON keys may contain delimiter characters.
 * @param {unknown} value @param {readonly string[]} keys @returns {boolean} */
function hasExactBuildFields(value, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {return false;}
  const actual = Object.keys(value).sort();
  return actual.length === keys.length && [...keys].sort().every((key, index) => key === actual[index]);
}

/** Admit an exact local file path before opening producer state or tools.
 * @param {string} file @param {string} label @param {number} [maximum] */
function canonicalBuildFile(file, label, maximum = 8 << 20) {
  if (typeof file !== "string") {throw new Error(label + " path is invalid");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.nlink !== 1 || stat.size < 1 || stat.size > maximum || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded canonical regular file: " + file);
  }
  return stat;
}

/** Bind the parsed value and its digest to a single opened file.
 * @param {string} file @param {number} maximum @param {string} label */
function readCanonicalBuildJson(file, maximum, label) {
  const identity = canonicalBuildFile(file, label, maximum), bytes = readBuildBytes(file, maximum, identity);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (!sameBuildFile(identity, canonicalBuildFile(file, label, maximum)) || fileHash(file) !== sha256) {
    throw new Error(label + " changed during admission");
  }
  let value;
  try {value = JSON.parse(bytes.toString("utf8"));} catch (error) {throw new Error(label + " is invalid JSON", {cause: error});}
  return {value, sha256, identity};
}

/** @param {string} directory @param {string} label @param {boolean} [allowMissing] */
function canonicalBuildDirectory(directory, label, allowMissing = false) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat && allowMissing) {return false;}
  if (!stat?.isDirectory() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is not a canonical directory: " + directory);
  }
  return true;
}

/** Stop at the first unrecognized entry so an unexpected tree cannot exhaust memory.
 * @param {string} directory @param {readonly string[]} expected */
function hasBuildDirectoryEntries(directory, expected) {
  const remaining = new Set(expected);
  if (remaining.size !== expected.length) {return false;}
  const handle = fs.opendirSync(directory);
  try {
    for (let entry; (entry = handle.readSync()) !== null;) {
      if (!remaining.delete(entry.name)) {return false;}
    }
  } finally {handle.closeSync();}
  return remaining.size === 0;
}

/** @param {string} directory @param {readonly string[]} expected @param {string} label */
function verifyBuildDirectoryEntries(directory, expected, label) {
  if (!hasBuildDirectoryEntries(directory, expected)) {
    throw new Error(label + " is incomplete or contains unknown entries; preserve it for inspection");
  }
}

/** Re-admit a completed build using the digest of the same exact bytes that were compared.
 * @param {string} file @param {Buffer} expected @param {string} label */
function verifyStableBuildBytes(file, expected, label) {
  const before = canonicalBuildFile(file, label);
  const stale = label + " is stale or changed; preserve it for inspection";
  if (before.size !== expected.length) {throw new Error(stale);}
  const bytes = readBuildBytes(file, before.size, before);
  const after = canonicalBuildFile(file, label);
  if (!sameBuildFile(before, after) || !bytes.equals(expected)) {
    throw new Error(stale);
  }
  return createHash("sha256").update(bytes).digest("hex");
}

/** @param {string} file @param {unknown} expected @param {string} label */
function verifyStableBuildJson(file, expected, label) {
  return verifyStableBuildBytes(file, Buffer.from(JSON.stringify(expected, null, 2) + "\n"), label);
}

/** One literal parser for compiler -MD and Gyp's deps_* := form. Never evaluate Make text;
 * escaped spaces stay in filenames, SYCL host/device rules are unioned, -MP rules add no deps.
 * NMake (-MV) quotes paths containing spaces and preserves Windows backslashes verbatim.
 * @param {string} text @param {string} base @param {"make" | "nmake"} [format] @returns {string[] | null} */
function buildDependencies(text, base, format = "make") {
  if (text.length > (4 << 20) || !path.isAbsolute(base) || !["make", "nmake"].includes(format)) {return null;}
  const unfolded = text.replace(/\\\r?\n/g, " "), gyp = unfolded.match(/^deps_[^\n]*?\s*:=([^\n]*)/m);
  if (gyp && format !== "make") {return null;}
  const bodies = gyp ? [gyp[1] || ""] : unfolded.trim().split(/\r?\n/).filter(line => !/^cmd_[^\n]*?\s*:=/.test(line))
    .map(line => line.match(/^(?:[A-Za-z]:)?(?:\\.|[^:\n])+:(?!=)[ \t]*(.*)$/)?.[1]);
  if (bodies.some(body => body === undefined || /[$\0]|(?:^|[^\\])#/.test(body))) {return null;}
  if (format === "nmake") {
    const body = bodies.join(" "), tokens = [], expression = /"([^"\r\n]+)"|([^\s"]+)/g;
    let end = 0, match;
    while ((match = expression.exec(body)) !== null) {
      if ((match.index === end && end !== 0) || /\S/.test(body.slice(end, match.index))) {return null;}
      tokens.push(match[1] || match[2] || ""); end = expression.lastIndex;
    }
    if (!tokens.length || tokens.length > 65536 || /\S/.test(body.slice(end))) {return null;}
    return [...new Set(tokens.map(token => path.resolve(base, token)))];
  }
  const tokens = bodies.join(" ").match(/(?:\\.|[^\s])+/g) || [];
  if (!tokens.length || tokens.length > 65536) {return null;}
  return [...new Set(tokens.map(token => path.resolve(base, token.replace(/\\(.)/g, "$1"))))];
}

/** Capture only explicitly selected files; no SDK/build-directory traversal.
 * @param {string[]} files @returns {Record<string, string>} */
function buildFileInputs(files) {
  if (files.length > 65536 || files.some(file => !path.isAbsolute(file) || /[\r\n\0]/.test(file))) {throw new Error("SDK build inputs must be bounded absolute file paths");}
  return Object.fromEntries([...new Set(files)].sort().map(file => [file, fileHash(file)]));
}

/** Freeze newly discovered SDK headers after compilation, rejecting inputs changed mid-build.
 * MoM source membership is already covered by sourceState; generated output is covered by its
 * source recipe. Resolve aliases before classification so an output/include symlink cannot
 * accidentally hide real SDK headers. Compiler/SDK prerequisites are checked before and after.
 * @param {string[]} depfiles @param {string} repo @param {string} output @param {number} startedAt
 * @param {Record<string, string>} initial */
function completeBuildInputs(depfiles, repo, output, startedAt, initial) {
  const files = new Set(Object.keys(initial)), roots = [repo, output].map(root => fs.realpathSync(root) + path.sep);
  for (const file of depfiles) {
    const deps = buildDependencies(readBuildText(file, 4 << 20), repo);
    if (!deps?.length) {throw new Error("Incomplete native compiler dependency graph");}
    for (const dependency of deps) {
      const real = fs.realpathSync(dependency);
      if (!roots.some(root => real.startsWith(root))) {files.add(real);}
    }
  }
  const current = buildFileInputs([...files]);
  for (const [file, hash] of Object.entries(initial)) {if (current[file] !== hash) {throw new Error("Native SDK input changed during compilation: " + file);}}
  for (const file of files) {
    if (!Object.hasOwn(initial, file) && fs.statSync(file).ctimeMs > startedAt) {throw new Error("New native SDK dependency changed during compilation: " + file);}
  }
  return current;
}

module.exports = {compilerIncludePaths, containsBuildPath, buildPathsOverlap, isBuildIncludeDirectory, nativeBuildJobs, fileHash, sortedBuildEntries, walk, sourceInputs, nativeSourceInputs, verifySourceInputs, readBuildBytes, readBuildJson, readBuildText,
  hasExactBuildFields, requireBuildFields, canonicalBuildFile, readBuildJsonSnapshot, readCanonicalBuildJson, canonicalBuildDirectory, hasBuildDirectoryEntries,
  verifyBuildDirectoryEntries, verifyStableBuildBytes, verifyStableBuildJson,
  buildDependencies, buildFileInputs, completeBuildInputs};
