"use strict";

// Original libuv/zlib sources serve both native release platforms. No Node runtime,
// shared application library, package-manager state or miner execution is involved.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive} = require("./native-audit");
const repo = path.resolve(__dirname, "..");
const ioPins = Object.freeze({
  libuv: Object.freeze({directory: "libuv-v1.51.0", sha256: "5f0557b90b1106de71951a3c3931de5e0430d78da1d9a10287ebc7a3f78ef8eb"}),
  zlib: Object.freeze({directory: "zlib-1.3.1", sha256: "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"}),
});
/** Literal CMake arguments cannot contain list separators, interpolation or controls.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
  !/[;"$]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
/** @typedef {{cmake: string, cc: string, ninja: string}} IoTools */

/** Build the same required static targets on Linux/Windows; do not build examples/tests
 * or rely on PATH to select a compiler/build runner. @param {string} output
 * @param {IoTools} tools @param {number} jobs @param {NodeJS.Platform} [platform] */
function nativeIoPlan(output, tools, jobs, platform = process.platform) {
  if (!["linux", "win32"].includes(platform) || !tools || Object.keys(tools).sort().join(",") !== "cc,cmake,ninja" ||
      ![output, ...Object.values(tools)].every(literal) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Native I/O SDK needs Linux/Windows, absolute tools/output and jobs in 1..16");
  }
  const windows = platform === "win32", cmakePath = (/** @type {string} */ p) => p.replaceAll("\\", "/");
  return Object.entries(ioPins).map(([name, pin]) => {
    const source = path.join(output, pin.directory), build = path.join(output, name + "-build");
    const target = name === "libuv" ? "uv_a" : "zlibstatic";
    const archive = path.join(build, windows ? name === "libuv" ? "libuv.lib" : "zlibstatic.lib" : name === "libuv" ? "libuv.a" : "libz.a");
    // libuv's target is uv_a, but its original Windows OUTPUT_NAME/PREFIX emit libuv.lib.
    // Keep the staged SDK name stable for existing source-bound consumers.
    const installedArchive = path.join(output, "lib", windows ? target + ".lib" : path.basename(archive));
    return {name, pin, source, build, archive, installedArchive, configure: [tools.cmake, "-S", source, "-B", build, "-G", "Ninja",
      "-DCMAKE_BUILD_TYPE:STRING=Release", "-DCMAKE_C_COMPILER:FILEPATH=" + cmakePath(tools.cc),
      "-DCMAKE_MAKE_PROGRAM:FILEPATH=" + cmakePath(tools.ninja), "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON",
      "-DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW", "-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5",
      ...(windows ? ["-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded"] : []),
      ...(name === "libuv" ? ["-DLIBUV_BUILD_SHARED:BOOL=OFF", "-DLIBUV_BUILD_TESTS:BOOL=OFF", "-DLIBUV_BUILD_BENCH:BOOL=OFF"] : ["-DZLIB_BUILD_EXAMPLES:BOOL=OFF"])],
    compile: [tools.cmake, "--build", build, "--parallel", String(jobs), "--target", target]};
  });
}

/** Accept only this pinned cohort's relative members before extracting into a new private
 * output. A matching hash is required too; this is not a general untrusted archive loader.
 * @param {string} listing @param {string} directory */
function validateIoArchive(listing, directory) {
  const rows = listing.trimEnd().split(/\r?\n/);
  if (!rows.length || rows.length > 10000 || rows.some(row => row !== directory && !row.startsWith(directory + "/") ||
      row.includes("\\") || row.split("/").some(part => part === "." || part === "..") ||
      Array.from(row).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127))) {
    throw new Error("Invalid pinned I/O source archive members");
  }
}

/** Small header-only traversal, never dependency/build trees. Reject symlink inputs and
 * preserve exact bytes in the staged SDK. @param {string} source @param {string} target */
function ioHeaders(source, target) {
  let count = 0;
  const copy = (/** @type {string} */ from, /** @type {string} */ to) => {
    for (const entry of fs.readdirSync(from, {withFileTypes: true})) {
      if (++count > 256 || entry.isSymbolicLink()) {throw new Error("Invalid bounded I/O header tree");}
      const input = path.join(from, entry.name), output = path.join(to, entry.name);
      if (entry.isDirectory()) {fs.mkdirSync(output); copy(input, output);} else if (entry.isFile() && entry.name.endsWith(".h")) {
        fs.copyFileSync(input, output, fs.constants.COPYFILE_EXCL);
      } else {throw new Error("Unexpected I/O SDK header entry");}
    }
  };
  copy(source, target);
}

/** Inventory only the tiny staged SDK, rejecting added shadow headers and symlinks too.
 * Bound traversal before hashing; never walk compiler/source/build trees. @param {string} root */
function ioFiles(root) {
  const files = /** @type {Record<string, string>} */ ({}); let count = 0;
  const visit = (/** @type {string} */ directory) => {
    if (!fs.lstatSync(directory).isDirectory()) {throw new Error("Invalid native I/O SDK directory");}
    const stream = fs.opendirSync(directory);
    try {
      for (let entry; (entry = stream.readSync());) {
        if (++count > 256 || entry.isSymbolicLink()) {throw new Error("Invalid bounded native I/O SDK tree");}
        const file = path.join(directory, entry.name);
        if (entry.isDirectory()) {visit(file);} else if (entry.isFile()) {files[path.relative(root, file).replaceAll("\\", "/")] = fileHash(file);} else {
          throw new Error("Native I/O SDK member is not a regular file");
        }
      }
    } finally {stream.closeSync();}
  };
  for (const name of ["include", "lib"]) {visit(path.join(root, name));}
  return files;
}

/** Produce reproducible static I/O inputs from already-downloaded checksum-pinned original
 * archives. Compiler logs/receipts stay outside the final installed application closure.
 * @param {{libuv: string, zlib: string}} archives @param {string} output @param {IoTools} tools
 * @param {number} jobs @param {NodeJS.ProcessEnv} [environment] */
function buildNativeIo(archives, output, tools, jobs, environment = process.env) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("I/O SDK production requires MOM_AOT_BUILD_ONLY=1");}
  const plan = nativeIoPlan(output, tools, jobs), env = nativeSdkEnvironment(environment);
  if (fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("I/O SDK output must be new");}
  if (!archives || Object.keys(archives).sort().join(",") !== "libuv,zlib") {throw new Error("Select both pinned I/O archives");}
  const inputs = Object.fromEntries(Object.entries(archives).map(([name, file]) => {
    const pin = Reflect.get(ioPins, name);
    if (!literal(file) || !fs.lstatSync(file).isFile() || fileHash(file) !== pin.sha256) {throw new Error("Pinned I/O source hash differs: " + name);}
    return [file, pin.sha256];
  }));
  const fingerprints = Object.fromEntries(Object.values(tools).map(file => [file, fileHash(file)]));
  const recipe = sourceInputs(repo, [], ["scripts/build-native-io.js", "scripts/build-environment.js", "scripts/build-source-inputs.js", "scripts/native-audit.js"]);
  fs.mkdirSync(output, {mode: 0o700}); fs.mkdirSync(path.join(output, "logs"));
  const run = (/** @type {string[]} */ command, /** @type {string} */ label, capture = false) => {
    const program = command[0]; if (!program) {throw new Error("Missing I/O SDK command");}
    const log = fs.openSync(path.join(output, "logs", label + ".log"), "wx");
    try {
      const result = spawnSync(program, command.slice(1), {cwd: output, env, encoding: "utf8", stdio: capture ? ["ignore", "pipe", log] : ["ignore", log, log], timeout: 15 * 60 * 1000, maxBuffer: 2 << 20});
      if (result.status !== 0 || result.error || result.signal) {throw new Error("Native I/O SDK stage failed: " + label + "; inspect " + output);}
      return result.stdout || "";
    } finally {fs.closeSync(log);}
  };
  fs.mkdirSync(path.join(output, "include")); fs.mkdirSync(path.join(output, "lib"));
  for (const row of plan) {
    const archive = Reflect.get(archives, row.name);
    validateIoArchive(run([tools.cmake, "-E", "tar", "tf", archive], row.name + "-list", true), row.pin.directory);
    run([tools.cmake, "-E", "tar", "xf", archive], row.name + "-extract");
    run(row.configure, row.name + "-configure"); run(row.compile, row.name + "-compile");
    if (!isStaticArchive(row.archive)) {throw new Error("Native I/O output is not a regular static archive");}
    fs.copyFileSync(row.archive, row.installedArchive, fs.constants.COPYFILE_EXCL);
    if (row.name === "libuv") {ioHeaders(path.join(row.source, "include"), path.join(output, "include"));} else {
      for (const [root, name] of [[row.source, "zlib.h"], [row.build, "zconf.h"]]) {
        if (root && name) {fs.copyFileSync(path.join(root, name), path.join(output, "include", name), fs.constants.COPYFILE_EXCL);}
      }
    }
  }
  for (const [file, hash] of Object.entries({...inputs, ...fingerprints})) {if (fileHash(file) !== hash) {throw new Error("Native I/O SDK inputs changed during build");}}
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Native I/O SDK producer");
  const artifacts = ioFiles(output);
  const result = {version: 1, target: process.platform, pins: ioPins, inputs, fingerprints, recipe, artifacts, buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Original static libuv/zlib SDK built; no application execution.");
  return result;
}

/** Resolve and hash every staged file before a consumer can compile; no -l lookup through
 * ambient system library paths. @param {string} root @param {NodeJS.Platform} [platform] */
function nativeIoInputs(root, platform = process.platform) {
  if (!literal(root)) {throw new Error("Select an absolute native I/O SDK root");}
  root = fs.realpathSync(root);
  const receipt = path.join(root, "result.json"), receiptSha256 = fileHash(receipt), record = readBuildJson(receipt);
  if (record.version !== 1 || record.target !== platform || record.buildOnly !== true || !isDeepStrictEqual(record.pins, ioPins) ||
      !record.artifacts || typeof record.artifacts !== "object" || Array.isArray(record.artifacts) || Object.keys(record.artifacts).length > 256) {
    throw new Error("Invalid native I/O SDK receipt");
  }
  for (const [name, hash] of Object.entries(record.artifacts)) {
    if (!/^(?:include\/[A-Za-z0-9_./-]+\.h|lib\/[A-Za-z0-9_.-]+\.(?:a|lib))$/.test(name) || name.split("/").some(part => part === "." || part === "..")) {throw new Error("Invalid native I/O SDK member");}
    const file = path.join(root, name);
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file || fileHash(file) !== hash) {throw new Error("Native I/O SDK bytes changed: " + name);}
  }
  const libraries = (platform === "win32" ? ["uv_a.lib", "zlibstatic.lib"] : ["libuv.a", "libz.a"]).map(name => path.join(root, "lib", name));
  for (const file of [...libraries, ...["uv.h", "zlib.h", "zconf.h"].map(name => path.join(root, "include", name))]) {
    if (!Object.hasOwn(record.artifacts, path.relative(root, file).replaceAll("\\", "/"))) {throw new Error("Incomplete native I/O SDK");}
  }
  if (!isDeepStrictEqual(record.artifacts, ioFiles(root))) {throw new Error("Native I/O SDK membership or bytes changed");}
  if (libraries.some(file => !isStaticArchive(file))) {throw new Error("Native I/O SDK libraries must be static archives");}
  if (fileHash(receipt) !== receiptSha256) {throw new Error("Native I/O SDK receipt changed during admission");}
  return {include: path.join(root, "include"), libraries, artifacts: record.artifacts, receipt, receiptSha256};
}

module.exports = {ioPins, nativeIoPlan, validateIoArchive, buildNativeIo, nativeIoInputs};
if (require.main === module) {
  try {
    const [libuv, zlib, output, cmake, cc, ninja, jobs = "2", extra] = process.argv.slice(2);
    if (!libuv || !zlib || !output || !cmake || !cc || !ninja || extra) {throw new Error("Usage: build-native-io.js /libuv.tar.gz /zlib.tar.gz /new/sdk /cmake /cc /ninja [jobs]");}
    buildNativeIo({libuv, zlib}, output, {cmake, cc, ninja}, Number(jobs));
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
