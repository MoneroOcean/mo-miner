"use strict";

// Build only the pinned LLVM OpenMP host runtime needed by AdaptiveCpp.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive, nativeDependencies} = require("./native-audit");
const {archiveName, archiveSha256, tarTool, verifyArchiveCatalog} = require("./native-llvm-sdk");
const {verifyBuildArchives} = require("./fetch-native-sdk-sources");

const repo = path.resolve(__dirname, ".."), sourceRootName = "llvm-project-21.1.8.src";
/** @typedef {"cc" | "cmake" | "cxx" | "ninja" | "readelf" | "tar"} ToolName */
/** @typedef {Record<ToolName, string>} ToolPaths */
/** @typedef {Record<ToolName, {path: string, sha256: string}>} ToolHashes */
/** @typedef {Record<ToolName, {firstLine: string, outputSha256: string}>} ToolVersions */
/** @typedef {{path: string, sha256: string, bytes: number}} FileArtifact */
/** @typedef {{path: string, sha256: string, bytes: number, argvSha256: string}} StageLog */
/** @typedef {Record<string, any>} JsonRecord */
/** @typedef {{version: number, jobs: number, tools: ToolPaths}} OpenmpConfig */
/** @typedef {ReturnType<typeof openmpSdkPlan>} OpenmpPlan */
/** @typedef {{version: number, role: string, platform: string, output: string,
 * archive: {path: string, sha256: string, bytes: number, catalog: unknown, selectedCatalog: unknown},
 * fetchReceipt: {path: string, sha256: string}, source: {path: string, files: number, entries: number, bytes: number, sha256: string},
 * config: {path: string, sha256: string}, tools: ToolHashes, toolVersions: ToolVersions, recipe: ReturnType<typeof sourceInputs>,
 * plan: OpenmpPlan, buildOutput: FileArtifact, sdk: FileArtifact,
 * smoke: {source: {path: string, sha256: string}, header: {path: string, sha256: string}, object: FileArtifact, binary: FileArtifact,
 *   run: {args: string[], status: number, signal: string | null, env: Record<string, string>, logSha256: string}, dynamicDependencies: string[]},
 * logs: Record<string, StageLog>, buildOnly: boolean, compiled: boolean, executed: boolean, publicationQualified: boolean}} OpenmpReceipt */
/** @type {readonly ToolName[]} */
const toolNames = Object.freeze(["cc", "cmake", "cxx", "ninja", "readelf", "tar"]);
const sourceLeaves = Object.freeze(["scripts/build-native-openmp-sdk.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
  "scripts/native-audit.js", "scripts/native-llvm-sdk.js"]);
// Acquisition is admitted separately by exact archive pins and immutable receipt identity.
// Only compilation policy belongs in this recipe; unrelated downloader edits do not change libomp.
function openmpBuildRecipe() {return sourceInputs(repo, [], [...sourceLeaves]);}
const sourceEnvironment = Object.freeze({PYTHONDONTWRITEBYTECODE: "1"});
const sourceByteLimit = 1 << 30, sourceFileLimit = 100000, sourceEntryLimit = 200000,
  archiveMemberLimit = 250000, archiveBytesLimit = 192 << 20, archiveOutputLimit = 128 << 20, noticeBytesLimit = 8 << 20,
  logBytesLimit = 128 << 20, producerDeadlineMs = 10 * 60 * 1000;
const smokeSource = "#include <omp.h>\nint main(void) {\n  int workers = 0;\n#pragma omp parallel num_threads(4) reduction(+:workers)\n  workers += 1;\n  return workers == 4 ? 0 : 1;\n}\n";
/** @param {unknown} value @returns {value is string} */
const isHash = value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);

/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !/[;"'$\\\r\n\0\x60]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} left @param {string} right */
function overlaps(left, right) {
  /** @param {string} parent @param {string} child */
  const within = (parent, child) => {
    const relative = path.relative(parent, child);
    return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
  };
  return within(left, right) || within(right, left);
}

/** @param {unknown} value @returns {ToolPaths} */
function validateTools(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== toolNames.join(",")) {
    throw new Error("Static OpenMP SDK needs the exact explicit Linux tool set");
  }
  const tools = /** @type {ToolPaths} */ (value);
  if (Object.values(tools).some(file => !absolute(file)) || path.basename(tools.cc || "") !== "icx" ||
      path.basename(tools.cxx || "") !== "icpx" || !tools.cc.startsWith("/opt/intel/oneapi/compiler/2026.2/") ||
      !tools.cxx.startsWith("/opt/intel/oneapi/compiler/2026.2/") || path.basename(tools.cmake || "") !== "cmake" ||
      path.dirname(tools.cmake || "") !== "/usr/bin" || path.basename(tools.ninja || "") !== "ninja" || path.dirname(tools.ninja || "") !== "/usr/bin" ||
      !["readelf", "x86_64-linux-gnu-readelf"].includes(path.basename(tools.readelf || "")) ||
      path.dirname(tools.readelf || "") !== "/usr/bin" || tools.tar !== tarTool) {
    throw new Error("Static OpenMP SDK needs canonical CMake/Ninja/Intel 2026.2/readelf/tar tools");
  }
  return tools;
}

/** Pure command plan; the producer deliberately builds one static host target with two jobs.
 * @param {string} archive @param {string} output @param {unknown} tools @param {number} [jobs] @param {string} [platform]
 * @returns {{version: number, platform: string, jobs: number, output: string, source: string, build: string, sdk: string, smoke: string,
 * buildArchive: string, installArchive: string, header: string, smokeC: string, smokeObject: string, smokeBinary: string,
 * extract: string[], configure: string[], buildCommand: string[], smokeCompile: string[], smokeLink: string[], smokeReadelf: string[],
 * cache: Record<string, string>, sourceEnvironment: Record<string, string>, smokeRun: string[]}} */
function openmpSdkPlan(archive, output, tools, jobs = 2, platform = process.platform) {
  if (platform !== "linux" || process.arch !== "x64" || !absolute(archive) || path.basename(archive) !== archiveName ||
      !absolute(output) || output.includes("=") || path.normalize(output) !== output || !Number.isInteger(jobs) || jobs !== 2) {
    throw new Error("Static OpenMP SDK needs the pinned LLVM archive, a canonical Linux x64 output, and exactly two jobs");
  }
  const selected = validateTools(tools), source = path.join(output, "source"), build = path.join(output, "build"),
    sdk = path.join(output, "sdk"), smoke = path.join(output, "smoke"), buildArchive = path.join(build, "runtime/src/libomp.a"),
    installArchive = path.join(sdk, "libomp.a"), header = path.join(build, "runtime/src/omp.h"),
    smokeC = path.join(smoke, "openmp-smoke.c"), smokeObject = path.join(smoke, "openmp-smoke.o"), smokeBinary = path.join(smoke, "openmp-smoke");
  const extract = [selected.tar, "-xJf", archive, "--directory", source, "--strip-components=1", "--no-same-owner", "--no-same-permissions", "--",
    sourceRootName + "/openmp/", sourceRootName + "/cmake/"];
  // Preserve diagnostics without embedding the temporary SDK prefix in debug data or __FILE__.
  // CMake parses these flag strings; quote spaces and reject '=' in the map's source prefix above.
  const prefixMap = "-ffile-prefix-map=" + JSON.stringify(output + "=/mom/openmp-sdk"),
    compileFlags = "-fno-builtin -ffunction-sections -fdata-sections " + prefixMap;
  const definitions = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=" + sdk,
    "-DCMAKE_C_COMPILER=" + selected.cc, "-DCMAKE_CXX_COMPILER=" + selected.cxx, "-DCMAKE_ASM_COMPILER=" + selected.cc,
    // Match the release application's section GC so unused upstream helper functions need no extra runtime.
    "-DCMAKE_C_FLAGS=" + compileFlags, "-DCMAKE_CXX_FLAGS=" + compileFlags, "-DCMAKE_ASM_FLAGS=" + prefixMap,
    "-DCMAKE_MAKE_PROGRAM=" + selected.ninja, "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DBUILD_TESTING=OFF", "-DLIBOMP_USE_QUAD_PRECISION=ON",
    "-DOPENMP_ENABLE_LIBOMPTARGET=OFF", "-DOPENMP_ENABLE_OMPT_TOOLS=OFF",
    "-DLIBOMP_ENABLE_SHARED=OFF", "-DLIBOMP_OMPT_SUPPORT=OFF", "-DLIBOMP_OMPD_SUPPORT=OFF", "-DLIBOMP_USE_ITT_NOTIFY=OFF",
    "-DLIBOMP_USE_HWLOC=OFF"];
  const configure = [selected.cmake, "-G", "Ninja", "-S", path.join(source, "openmp"), "-B", build, ...definitions];
  const buildCommand = [selected.cmake, "--build", build, "--target", "omp", "--parallel", "2", "--verbose"];
  const smokeCompile = [selected.cc, "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", "-ffreestanding", "-fno-builtin", "-fopenmp", "-I", path.dirname(header),
    "-c", smokeC, "-o", smokeObject];
  // Mirror the release link's GC; keep -fopenmp off so icx cannot select shared Intel OpenMP.
  const smokeLink = [selected.cc, "-no-intel-lib", "-pie", "-Wl,--gc-sections", smokeObject, buildArchive,
    "-pthread", "-ldl", "-lm", "-Wl,--no-undefined", "-o", smokeBinary];
  const smokeReadelf = [selected.readelf, "--dynamic", "--wide", smokeBinary];
  const cache = {
    BUILD_TESTING: "OFF", CMAKE_ASM_COMPILER: selected.cc, CMAKE_BUILD_TYPE: "Release", CMAKE_C_COMPILER: selected.cc,
    CMAKE_CXX_COMPILER: selected.cxx, CMAKE_CXX_FLAGS: compileFlags, CMAKE_C_FLAGS: compileFlags, CMAKE_ASM_FLAGS: prefixMap,
    CMAKE_EXPORT_COMPILE_COMMANDS: "ON", CMAKE_INSTALL_PREFIX: sdk,
    CMAKE_INTERPROCEDURAL_OPTIMIZATION: "OFF", CMAKE_MAKE_PROGRAM: selected.ninja, CMAKE_POSITION_INDEPENDENT_CODE: "ON",
    LIBOMP_USE_QUAD_PRECISION: "ON",
    LIBOMP_ENABLE_SHARED: "OFF", LIBOMP_OMPD_SUPPORT: "OFF", LIBOMP_OMPT_SUPPORT: "OFF", LIBOMP_USE_HWLOC: "OFF",
    LIBOMP_USE_ITT_NOTIFY: "OFF", OPENMP_ENABLE_LIBOMPTARGET: "OFF", OPENMP_ENABLE_OMPT_TOOLS: "OFF",
  };
  return {version: 1, platform, jobs, output, source, build, sdk, smoke, buildArchive, installArchive, header, smokeC, smokeObject, smokeBinary,
    extract, configure, buildCommand, smokeCompile, smokeLink, smokeReadelf, cache, sourceEnvironment: {...sourceEnvironment}, smokeRun: [smokeBinary]};
}

/** Prevent source-side Python imports from adding bytecode to the pinned extraction.
 * @param {NodeJS.ProcessEnv} environment @returns {NodeJS.ProcessEnv} */
function openmpSdkEnvironment(environment) {
  return {...nativeSdkEnvironment(environment), ...sourceEnvironment};
}

/** Verify the complete member listing; the shared LLVM catalog validator independently pins all archive roots.
 * @param {string} listing */
function validateOpenmpArchiveListing(listing) {
  const rows = listing.split(/\r?\n/); if (rows.at(-1) === "") {rows.pop();}
  if (!rows.length || rows.length > archiveMemberLimit) {throw new Error("Invalid LLVM OpenMP archive member listing");}
  const seen = new Set(), selected = [];
  for (const row of rows) {
    const name = row.endsWith("/") ? row.slice(0, -1) : row, parts = name.split("/");
    const isSourceRootDirectory = row === sourceRootName + "/";
    if (!name || name.length > 4096 || /[\\\r\n\0]/.test(name) || Array.from(name).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) ||
        (!isSourceRootDirectory && !name.startsWith(sourceRootName + "/")) || parts.some(part => !part || part === "." || part === "..") || seen.has(name)) {
      throw new Error("Unsafe or duplicate LLVM OpenMP archive member");
    }
    seen.add(name);
    if (name === sourceRootName + "/openmp" || name.startsWith(sourceRootName + "/openmp/") ||
        name === sourceRootName + "/cmake" || name.startsWith(sourceRootName + "/cmake/")) {selected.push(row);}
  }
  for (const root of [sourceRootName + "/openmp/", sourceRootName + "/cmake/"]) {
    if (!rows.includes(root)) {throw new Error("Pinned LLVM source archive lacks selected OpenMP/CMake root: " + root);}
  }
  /** @param {string[]} values */
  const digest = values => createHash("sha256").update(values.join("\n") + "\n").digest("hex");
  return {members: rows.length, sha256: digest(rows), selectedMembers: selected.length, selectedSha256: digest(selected)};
}

/** Hash only the two extracted, bounded upstream roots; safe relative symlinks are recorded, never followed recursively.
 * Capture the regular top-level OpenMP license from this same inventory for downstream notices.
 * @param {string} root */
function sourceInventory(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root || !isDeepStrictEqual(fs.readdirSync(root).sort(), ["cmake", "openmp"])) {
    throw new Error("Static OpenMP source must contain only the selected cmake/openmp roots");
  }
  const hash = createHash("sha256"); let count = 0, entries = 0, bytes = 0;
  /** @type {{path: string, sha256: string, bytes: number} | undefined} */
  let license;
  /** @param {string} directory */
  function visit(directory) {
    const directoryStat = fs.lstatSync(directory);
    if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {throw new Error("Static OpenMP source contains a linked directory");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), relative = path.relative(root, file).split(path.sep).join("/");
      if (!relative || relative.length > 4096 || /[\\\r\n\0]/.test(relative) || Array.from(relative).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) || ++entries > sourceEntryLimit) {
        throw new Error("Static OpenMP source inventory has an invalid or excessive path");
      }
      if (entry.isDirectory()) {hash.update("directory\0" + relative + "\n"); visit(file); continue;}
      const child = fs.lstatSync(file);
      if (entry.isSymbolicLink() && child.isSymbolicLink()) {
        if (relative === "openmp/LICENSE.TXT") {throw new Error("Static OpenMP license must be a regular canonical file");}
        const target = fs.readlinkSync(file), resolved = path.resolve(path.dirname(file), target), relativeTarget = path.relative(root, resolved), real = fs.realpathSync(file),
          targetStat = fs.statSync(file);
        if (!target || target.length > 4096 || /[\r\n\0]/.test(target) || path.isAbsolute(target) || path.isAbsolute(relativeTarget) ||
            relativeTarget === ".." || relativeTarget.startsWith(".." + path.sep) || !(real === root || real.startsWith(root + path.sep)) ||
            (!targetStat.isFile() && !targetStat.isDirectory())) {
          throw new Error("Static OpenMP source contains an unsafe symlink: " + relative);
        }
        bytes += Buffer.byteLength(target); hash.update("link\0" + relative + "\0" + target + "\n"); continue;
      }
      if (!entry.isFile() || child.isSymbolicLink() || !child.isFile() || ++count > sourceFileLimit || (bytes += child.size) > sourceByteLimit) {
        throw new Error("Static OpenMP source is not a bounded regular-file tree");
      }
      const digest = fileHash(file);
      if (relative === "openmp/LICENSE.TXT") {
        if (child.size < 1 || child.size > noticeBytesLimit || fs.realpathSync(file) !== file) {
          throw new Error("Static OpenMP license must be bounded, regular and canonical");
        }
        license = {path: "source/openmp/LICENSE.TXT", sha256: digest, bytes: child.size};
      }
      hash.update("file\0" + relative + "\0" + child.size + "\0" + digest + "\n");
    }
  }
  for (const name of ["cmake", "openmp"]) {visit(path.join(root, name));}
  if (!count) {throw new Error("Static OpenMP source tree is empty");}
  if (!license) {throw new Error("Static OpenMP source is missing its top-level license");}
  return {count, entries, bytes, sha256: hash.digest("hex"), license};
}

/** Match the full admitted source inventory while retaining its receipt-backed license identity.
 * @param {string} root @param {OpenmpReceipt["source"]} expected */
function verifyOpenmpSourceInventory(root, expected) {
  const source = sourceInventory(root);
  if (!isDeepStrictEqual(expected, {path: "source", files: source.count, entries: source.entries, bytes: source.bytes, sha256: source.sha256})) {
    throw new Error("Static OpenMP source membership or bytes changed");
  }
  return source;
}

/** Recheck the inventory-derived license as a bounded canonical file before returning it downstream.
 * @param {string} root @param {{path: string, sha256: string, bytes: number}} notice */
function verifyOpenmpNoticeInput(root, notice) {
  if (!absolute(root) || fs.realpathSync(root) !== root || !notice || Object.keys(notice).sort().join(",") !== "bytes,path,sha256" ||
      notice.path !== "source/openmp/LICENSE.TXT" || !isHash(notice.sha256) || !Number.isSafeInteger(notice.bytes) ||
      notice.bytes < 1 || notice.bytes > noticeBytesLimit) {
    throw new Error("Invalid static OpenMP source license binding");
  }
  const file = path.join(root, notice.path), stat = regularInput(file, "source license");
  if (stat.size !== notice.bytes || fileHash(file) !== notice.sha256) {throw new Error("Static OpenMP source license changed");}
  return {file, notice};
}

/** Parse CMake's small cache and require the exact non-offload static host configuration.
 * @param {string} cacheFile @param {Record<string,string>} expected */
function verifyCMakeCache(cacheFile, expected) {
  const stat = fs.lstatSync(cacheFile);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > (8 << 20)) {throw new Error("Invalid or excessive OpenMP CMake cache");}
  const contents = fs.readFileSync(cacheFile, "utf8"), values = /** @type {Record<string, string>} */ (Object.create(null));
  for (const line of contents.split(/\r?\n/)) {
    const match = line.match(/^([^:#][^:]*):[^=]+=([^\r\n]*)$/);
    if (!match) {continue;}
    const name = match[1];
    if (name === undefined || Object.hasOwn(values, name)) {throw new Error("Duplicate OpenMP CMake cache variable");}
    values[name] = match[2] || "";
  }
  for (const [name, value] of Object.entries(expected)) {
    if (values[name] !== value) {throw new Error("OpenMP CMake cache differs from the static host recipe: " + name);}
  }
  return Object.fromEntries(Object.keys(expected).sort().map(name => {
    const value = values[name]; if (value === undefined) {throw new Error("Missing OpenMP CMake cache variable: " + name);}
    return [name, value];
  }));
}

/** @param {string} file @param {string} label @param {boolean} [executable] */
function regularInput(file, label, executable = false) {
  if (!absolute(file)) {throw new Error("OpenMP " + label + " must be an absolute path");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file) {throw new Error("OpenMP " + label + " must be a canonical regular file");}
  if (executable) {fs.accessSync(file, fs.constants.X_OK);}
  return stat;
}

/** Persisted tool reports must compare equal after JSON reload; only allowlisted tool keys are emitted.
 * @template T @param {(name: ToolName) => T} value @returns {Record<ToolName, T>} */
function toolRecord(value) {
  const record = /** @type {Record<ToolName, T>} */ ({});
  for (const name of toolNames) {record[name] = value(name);}
  return record;
}

/** @param {ToolPaths} tools @returns {ToolHashes} */
function toolIdentity(tools) {
  const selected = validateTools(tools);
  return toolRecord(name => {
    const file = selected[name]; regularInput(file, "tool " + name, true);
    return {path: file, sha256: fileHash(file)};
  });
}

/** @param {string} file @param {string[]} args @param {NodeJS.ProcessEnv} env @param {number} timeout */
function capture(file, args, env, timeout) {
  const result = spawnSync(file, args, {encoding: "utf8", env, timeout, maxBuffer: 1 << 20});
  if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || typeof result.stderr !== "string") {
    throw new Error("OpenMP tool command failed: " + path.basename(file), {cause: result.error});
  }
  const output = (result.stdout || "") + (result.stderr || "");
  if (!output.trim() || Buffer.byteLength(output) > (1 << 20)) {throw new Error("OpenMP tool version output is empty or excessive");}
  return {firstLine: output.trim().split(/\r?\n/, 1)[0] || "", outputSha256: createHash("sha256").update(output).digest("hex")};
}

/** @param {ToolPaths} tools @param {NodeJS.ProcessEnv} env @param {number} deadline @returns {ToolVersions} */
function toolVersions(tools, env, deadline) {
  const versions = toolRecord(name => {
    const remaining = deadline - Date.now(); if (remaining <= 0) {throw new Error("Static OpenMP producer deadline exceeded during tool verification");}
    return capture(tools[name], ["--version"], env, Math.min(30000, remaining));
  });
  const intel = /^Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler 2026\.2(?:\.|\s|$)/;
  if (!intel.test(versions.cc.firstLine) || !intel.test(versions.cxx.firstLine)) {
    throw new Error("Static OpenMP SDK requires the selected Intel 2026.2 icx/icpx toolchain");
  }
  if (!/^(?:GNU readelf|GNU Binutils)\b/.test(versions.readelf.firstLine)) {throw new Error("Static OpenMP SDK requires the selected GNU readelf");}
  return versions;
}

/** @param {number} deadline */
function remainingMs(deadline) {
  const remaining = deadline - Date.now();
  if (remaining <= 0) {throw new Error("Static OpenMP producer exceeded its ten-minute deadline");}
  return remaining;
}

/** @param {string[]} command @param {string} cwd @param {NodeJS.ProcessEnv} env @param {string} log @param {number} deadline */
function runLogged(command, cwd, env, log, deadline) {
  fs.writeFileSync(log + ".argv.json", JSON.stringify(command) + "\n", {flag: "wx", mode: 0o600});
  const fd = fs.openSync(log, "wx", 0o600);
  let result;
  try {
    result = spawnSync(command[0] ?? "", command.slice(1), {cwd, env, stdio: ["ignore", fd, fd], timeout: remainingMs(deadline)});
  } finally {
    fs.closeSync(fd);
  }
  const stat = fs.statSync(log);
  if (result?.error || result?.signal || result?.status !== 0 || stat.size > logBytesLimit) {
    throw new Error("Static OpenMP command failed or exceeded its evidence bound: " + path.basename(log), {cause: result?.error});
  }
  return {path: path.relative(path.dirname(path.dirname(log)), log).split(path.sep).join("/"), sha256: fileHash(log), bytes: stat.size,
    argvSha256: fileHash(log + ".argv.json")};
}

/** @param {string} archive @param {string} expectedHash
 * @param {{verify?: (sourceRoot: string) => {archives?: Record<string, {path: string, sha256: string, bytes: number}>}}} [readers] */
function verifyOpenmpArchiveInput(archive, expectedHash = archiveSha256, readers = {}) {
  if (!absolute(archive) || path.basename(archive) !== archiveName || expectedHash !== archiveSha256) {
    throw new Error("Static OpenMP SDK needs the exact pinned LLVM 21.1.8 archive path and digest");
  }
  const sourceRoot = path.dirname(archive), verifier = readers.verify || verifyBuildArchives, receipt = verifier(sourceRoot), row = receipt.archives?.["llvm"],
    stat = regularInput(archive, "original source archive");
  if (stat.size > archiveBytesLimit || !row || row.path !== archive || row.sha256 !== archiveSha256 || row.bytes !== stat.size || fileHash(archive) !== archiveSha256) {
    throw new Error("Static OpenMP source archive differs from the immutable pinned LLVM fetch receipt");
  }
  const fetchReceipt = path.join(sourceRoot, "fetch.json"), fetchReceiptSha256 = fileHash(fetchReceipt);
  return {sourceRoot, archive: {path: archive, sha256: archiveSha256, bytes: stat.size},
    fetchReceipt: {path: fetchReceipt, sha256: fetchReceiptSha256}, receipt};
}

/** @param {string} file @param {string} label @param {number} maximum */
function readText(file, label, maximum) {
  const stat = regularInput(file, label);
  if (!stat.size || stat.size > maximum) {throw new Error("OpenMP " + label + " is empty or excessive");}
  return fs.readFileSync(file, "utf8");
}

/** @param {string} file @param {string} label */
function verifyStaticOutput(file, label = "static library") {
  const stat = regularInput(file, label), digest = fileHash(file);
  if (stat.size < 8 || stat.size > archiveOutputLimit || !isStaticArchive(file)) {throw new Error("OpenMP output is not a bounded regular static archive");}
  return {path: file, sha256: digest, bytes: stat.size};
}

/** @param {string} readelfText */
function verifyOpenmpDynamicDependencies(readelfText) {
  return nativeDependencies(readelfText);
}

/** @param {unknown} value @returns {value is ReturnType<typeof sourceInputs>} */
function isSourceInputRecord(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== "files,graph") {return false;}
  const input = /** @type {JsonRecord} */ (value);
  return ["files", "graph"].every(name => {
    const rows = input[name];
    return rows !== null && typeof rows === "object" && !Array.isArray(rows) &&
      Object.entries(rows).every(([file, hash]) => file.length > 0 && isHash(hash));
  });
}

/** @param {unknown} record @returns {OpenmpReceipt} */
function assertOpenmpReceiptShape(record) {
  const keys = ["archive", "buildOnly", "buildOutput", "compiled", "config", "executed", "fetchReceipt", "logs", "output", "plan",
    "platform", "publicationQualified", "recipe", "role", "sdk", "smoke", "source", "toolVersions", "tools", "version"];
  if (!record || typeof record !== "object" || Array.isArray(record)) {throw new Error("Invalid or stale static OpenMP SDK receipt");}
  const receipt = /** @type {OpenmpReceipt} */ (record);
  if (Object.keys(receipt).sort().join(",") !== keys.sort().join(",") ||
      receipt.version !== 1 || receipt.role !== "linux-static-openmp-sdk" || receipt.platform !== "linux" || receipt.buildOnly !== true ||
      receipt.compiled !== true || receipt.executed !== true || receipt.publicationQualified !== false || !isSourceInputRecord(receipt.recipe) ||
      receipt.config?.path !== "tools.json" || !isHash(receipt.config.sha256) ||
      !isHash(receipt.archive?.sha256) || !isHash(receipt.fetchReceipt?.sha256) ||
      receipt.archive.sha256 !== archiveSha256 || receipt.smoke?.run?.status !== 0 || receipt.smoke.run.signal !== null) {
    throw new Error("Invalid or stale static OpenMP SDK receipt");
  }
  return /** @type {OpenmpReceipt} */ (receipt);
}

/** Build the pinned source archive with the signed Intel compiler and admit a CPU-only static-link smoke.
 * @param {string} archive @param {string} output @param {OpenmpConfig} config
 * @param {NodeJS.ProcessEnv} environment @param {string} configFile @param {string} configSha256 */
function buildNativeOpenmpSdk(archive, output, config, environment, configFile, configSha256) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1" || process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Static OpenMP SDK production requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  if (!config || typeof config !== "object" || Array.isArray(config) || Object.keys(config).sort().join(",") !== "jobs,tools,version" ||
      config.version !== 1 || config.jobs !== 2 || !absolute(configFile || "") || !isHash(configSha256 || "")) {
    throw new Error("Invalid static OpenMP tool descriptor");
  }
  const tools = validateTools(config.tools), selectedOutput = path.resolve(output), parent = path.dirname(selectedOutput),
    plan = openmpSdkPlan(archive, selectedOutput, tools, config.jobs, process.platform), parentStat = fs.lstatSync(parent, {throwIfNoEntry: false}),
    toolHashes = toolIdentity(tools), start = Date.now(), deadline = start + producerDeadlineMs, env = openmpSdkEnvironment(environment);
  if (!absolute(archive) || !absolute(output) || output !== selectedOutput || path.normalize(output) !== output ||
      !parentStat?.isDirectory() || parentStat.isSymbolicLink() || fs.realpathSync(parent) !== parent || fs.existsSync(selectedOutput) ||
      !path.relative(repo, selectedOutput) || !path.relative(repo, selectedOutput).startsWith(".." + path.sep) ||
      overlaps(selectedOutput, path.dirname(archive)) || overlaps(selectedOutput, configFile || "") ||
      [archive, configFile || "", ...Object.values(tools)].some(file => overlaps(selectedOutput, file))) {
    throw new Error("Static OpenMP output must be new, canonical and separate from all inputs/repository files");
  }
  regularInput(configFile || "", "tool descriptor");
  if (fs.realpathSync(configFile || "") !== configFile || fileHash(configFile || "") !== configSha256) {
    throw new Error("Static OpenMP tool descriptor changed before compilation");
  }
  const sourceInput = verifyOpenmpArchiveInput(archive), catalog = verifyArchiveCatalog(archive);
  const listing = spawnSync(tools.tar, ["-tf", archive, "--quoting-style=literal"],
    {encoding: "utf8", env, timeout: Math.min(60000, remainingMs(deadline)), maxBuffer: 64 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || listing.stderr || typeof listing.stdout !== "string") {
    throw new Error("Pinned LLVM OpenMP archive member listing failed", {cause: listing.error});
  }
  const selectedCatalog = validateOpenmpArchiveListing(listing.stdout), beforeArchive = fileHash(archive), versions = toolVersions(tools, env, deadline);
  if (beforeArchive !== archiveSha256 || fileHash(configFile || "") !== configSha256 ||
      toolNames.some(name => fileHash(tools[name]) !== toolHashes[name].sha256)) {
    throw new Error("Static OpenMP archive, tool descriptor, or compiler tools changed during preflight");
  }
  const recipe = openmpBuildRecipe();
  fs.mkdirSync(selectedOutput, {mode: 0o700});
  for (const directory of [plan.source, plan.build, plan.sdk, plan.smoke, path.join(selectedOutput, "logs")]) {
    fs.mkdirSync(directory, {mode: 0o700});
  }
  fs.writeFileSync(path.join(selectedOutput, "tools.json"), JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.writeFileSync(plan.smokeC, smokeSource, {flag: "wx", mode: 0o600});
  /** @param {string} name */
  const log = name => path.join(selectedOutput, "logs", name + ".log"),
    logs = /** @type {Record<string, StageLog>} */ (Object.create(null));
  logs["extract"] = runLogged(plan.extract, selectedOutput, env, log("extract"), deadline);
  const sourceBefore = sourceInventory(plan.source);
  logs["configure"] = runLogged(plan.configure, selectedOutput, env, log("configure"), deadline);
  const checkedCache = verifyCMakeCache(path.join(plan.build, "CMakeCache.txt"), plan.cache);
  logs["build"] = runLogged(plan.buildCommand, selectedOutput, env, log("build"), deadline);
  const buildOutput = verifyStaticOutput(plan.buildArchive, "build archive"), headerStat = regularInput(plan.header, "generated OpenMP header");
  if (headerStat.size < 1 || headerStat.size > (1 << 20)) {throw new Error("Generated OpenMP header is empty or excessive");}
  fs.copyFileSync(plan.buildArchive, plan.installArchive, fs.constants.COPYFILE_EXCL);
  const sdkOutput = verifyStaticOutput(plan.installArchive, "staged archive");
  if (buildOutput.sha256 !== sdkOutput.sha256 || buildOutput.bytes !== sdkOutput.bytes) {throw new Error("Staged OpenMP archive differs from its build output");}
  logs["smoke-compile"] = runLogged(plan.smokeCompile, selectedOutput, env, log("smoke-compile"), deadline);
  const smokeObjectStat = regularInput(plan.smokeObject, "CPU smoke object"), smokeObject = {path: plan.smokeObject,
    sha256: fileHash(plan.smokeObject), bytes: smokeObjectStat.size};
  logs["smoke-link"] = runLogged(plan.smokeLink, selectedOutput, env, log("smoke-link"), deadline);
  const smokeBinaryStat = regularInput(plan.smokeBinary, "CPU smoke executable", true), smokeBinary = {path: plan.smokeBinary,
    sha256: fileHash(plan.smokeBinary), bytes: smokeBinaryStat.size};
  logs["smoke-readelf"] = runLogged(plan.smokeReadelf, selectedOutput, env, log("smoke-readelf"), deadline);
  const dynamicDependencies = verifyOpenmpDynamicDependencies(readText(log("smoke-readelf"), "readelf evidence", 4 << 20)),
    smokeEnv = {...env, LD_LIBRARY_PATH: "/nonexistent", OMP_NUM_THREADS: "4", OMP_DYNAMIC: "FALSE", OMP_THREAD_LIMIT: "4"};
  logs["smoke-run"] = runLogged(plan.smokeRun, selectedOutput, smokeEnv, log("smoke-run"), deadline);
  const sourceAfter = sourceInventory(plan.source);
  if (sourceBefore.sha256 !== sourceAfter.sha256 || sourceBefore.entries !== sourceAfter.entries || sourceBefore.bytes !== sourceAfter.bytes) {
    throw new Error("Pinned OpenMP/CMake source changed during compilation or smoke test");
  }
  verifySourceInputs(recipe, openmpBuildRecipe(), "Static OpenMP SDK producer");
  if (fileHash(archive) !== archiveSha256 || fileHash(sourceInput.fetchReceipt.path) !== sourceInput.fetchReceipt.sha256 ||
      fileHash(configFile || "") !== configSha256 || toolNames.some(name => fileHash(tools[name]) !== toolHashes[name].sha256)) {
    throw new Error("Static OpenMP source archive, fetch receipt, tools or descriptor changed during production");
  }
  const record = {version: 1, role: "linux-static-openmp-sdk", platform: "linux", output: selectedOutput,
    archive: {...sourceInput.archive, catalog, selectedCatalog}, fetchReceipt: sourceInput.fetchReceipt,
    source: {path: "source", files: sourceAfter.count, entries: sourceAfter.entries, bytes: sourceAfter.bytes, sha256: sourceAfter.sha256},
    config: {path: "tools.json", sha256: fileHash(path.join(selectedOutput, "tools.json"))}, tools: toolHashes, toolVersions: versions,
    recipe, plan: {...plan, cache: checkedCache}, buildOutput, sdk: sdkOutput,
    smoke: {source: {path: "smoke/openmp-smoke.c", sha256: fileHash(plan.smokeC)},
      header: {path: path.relative(selectedOutput, plan.header), sha256: fileHash(plan.header)}, object: smokeObject, binary: smokeBinary,
      run: {args: plan.smokeRun, status: 0, signal: null,
        env: {LD_LIBRARY_PATH: "/nonexistent", OMP_NUM_THREADS: "4", OMP_DYNAMIC: "FALSE", OMP_THREAD_LIMIT: "4"},
        logSha256: logs["smoke-run"].sha256}, dynamicDependencies}, logs,
    buildOnly: true, compiled: true, executed: true, publicationQualified: false};
  fs.writeFileSync(path.join(selectedOutput, "result.json"), JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const admitted = nativeOpenmpInputs(selectedOutput);
  console.log("[native-sdk] Built and CPU-smoke-tested pinned static LLVM OpenMP host runtime; no offload targets or release qualification.");
  return {root: selectedOutput, ...admitted};
}

/** Re-admit the original source receipt, current compiler tools, source tree, static archive and smoke evidence.
 * @param {string} root */
function nativeOpenmpInputs(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root || overlaps(root, repo)) {throw new Error("Select a canonical external static OpenMP SDK root");}
  const receiptPath = path.join(root, "result.json"); regularInput(receiptPath, "producer receipt");
  const receiptSha256 = fileHash(receiptPath), record = assertOpenmpReceiptShape(readBuildJson(receiptPath));
  if (record.output !== root) {throw new Error("Static OpenMP SDK receipt does not bind its selected output root");}
  const tools = validateTools(Object.fromEntries(Object.entries(record.tools || {}).map(([name, row]) => [name, row?.path]))),
    currentTools = toolIdentity(tools), env = nativeSdkEnvironment(process.env), versions = toolVersions(tools, env, Date.now() + 2 * 60 * 1000);
  if (!isDeepStrictEqual(currentTools, record.tools) || !isDeepStrictEqual(versions, record.toolVersions)) {
    throw new Error("Static OpenMP SDK tool identity/version changed");
  }
  const plan = openmpSdkPlan(record.archive.path, root, tools, 2, "linux");
  if (!isDeepStrictEqual(record.plan, plan)) {throw new Error("Static OpenMP SDK command plan or path identity is stale");}
  const sourceRoot = path.dirname(record.archive.path), sourceInput = verifyOpenmpArchiveInput(record.archive.path), catalog = verifyArchiveCatalog(record.archive.path),
    listing = spawnSync(tools.tar, ["-tf", record.archive.path, "--quoting-style=literal"],
      {encoding: "utf8", env, timeout: 60000, maxBuffer: 64 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || listing.stderr || typeof listing.stdout !== "string") {
    throw new Error("Pinned LLVM OpenMP archive re-admission failed", {cause: listing.error});
  }
  const selectedCatalog = validateOpenmpArchiveListing(listing.stdout);
  if (record.archive.path !== path.join(sourceRoot, archiveName) || record.archive.sha256 !== archiveSha256 ||
      record.archive.bytes !== fs.statSync(record.archive.path).size || !isDeepStrictEqual(record.archive.catalog, catalog) ||
      !isDeepStrictEqual(record.archive.selectedCatalog, selectedCatalog) ||
      record.fetchReceipt.path !== path.join(sourceRoot, "fetch.json") || record.fetchReceipt.sha256 !== sourceInput.fetchReceipt.sha256) {
    throw new Error("Static OpenMP SDK receipt lost its pinned source/fetch provenance");
  }
  const toolsPath = path.join(root, "tools.json"); regularInput(toolsPath, "tool descriptor");
  const toolsConfig = readBuildJson(toolsPath);
  if (!isDeepStrictEqual(toolsConfig, {version: 1, jobs: 2, tools}) || fileHash(path.join(root, "tools.json")) !== record.config.sha256) {
    throw new Error("Static OpenMP SDK tool descriptor changed");
  }
  const recipe = openmpBuildRecipe(); verifySourceInputs(record.recipe, recipe, "Static OpenMP SDK producer");
  const source = verifyOpenmpSourceInventory(path.join(root, "source"), record.source);
  verifyCMakeCache(path.join(root, "build/CMakeCache.txt"), plan.cache);
  const buildOutput = verifyStaticOutput(plan.buildArchive, "build archive"), sdkOutput = verifyStaticOutput(plan.installArchive, "staged archive");
  if (!isDeepStrictEqual(record.buildOutput, buildOutput) || !isDeepStrictEqual(record.sdk, sdkOutput) ||
      buildOutput.sha256 !== sdkOutput.sha256 || buildOutput.bytes !== sdkOutput.bytes) {
    throw new Error("Static OpenMP archive output changed");
  }
  const smokeSourcePath = path.join(root, "smoke/openmp-smoke.c"), smokeObjectPath = plan.smokeObject, smokeBinaryPath = plan.smokeBinary,
    smokeSourceStat = regularInput(smokeSourcePath, "CPU smoke source"), smokeObjectStat = regularInput(smokeObjectPath, "CPU smoke object"),
    smokeBinaryStat = regularInput(smokeBinaryPath, "CPU smoke executable", true);
  if (smokeSourceStat.size > 4096 || fs.readFileSync(smokeSourcePath, "utf8") !== smokeSource ||
      record.smoke.source.path !== "smoke/openmp-smoke.c" || record.smoke.source.sha256 !== fileHash(smokeSourcePath) ||
      record.smoke.header.path !== path.relative(root, plan.header) || record.smoke.header.sha256 !== fileHash(plan.header) ||
      record.smoke.object.path !== smokeObjectPath || record.smoke.object.bytes !== smokeObjectStat.size ||
      record.smoke.object.sha256 !== fileHash(smokeObjectPath) || record.smoke.binary.path !== smokeBinaryPath ||
      record.smoke.binary.bytes !== smokeBinaryStat.size || record.smoke.binary.sha256 !== fileHash(smokeBinaryPath) ||
      !isDeepStrictEqual(record.smoke.run.args, plan.smokeRun) ||
      !isDeepStrictEqual(record.smoke.run.env, {LD_LIBRARY_PATH: "/nonexistent", OMP_NUM_THREADS: "4", OMP_DYNAMIC: "FALSE", OMP_THREAD_LIMIT: "4"})) {
    throw new Error("Static OpenMP CPU smoke artifact/command receipt is stale");
  }
  const readelfText = readText(path.join(root, "logs/smoke-readelf.log"), "readelf evidence", 4 << 20),
    dynamicDependencies = verifyOpenmpDynamicDependencies(readelfText);
  if (!isDeepStrictEqual(record.smoke.dynamicDependencies, dynamicDependencies) || record.smoke.run.logSha256 !== record.logs?.["smoke-run"]?.sha256) {
    throw new Error("Static OpenMP CPU smoke dynamic/runtime evidence changed");
  }
  const expectedCommands = /** @type {Record<string, string[]>} */ ({extract: plan.extract, configure: plan.configure, build: plan.buildCommand,
      "smoke-compile": plan.smokeCompile, "smoke-link": plan.smokeLink, "smoke-readelf": plan.smokeReadelf, "smoke-run": plan.smokeRun}),
    expectedLogs = Object.keys(expectedCommands).sort(), logs = record.logs;
  if (!logs || Object.keys(logs).sort().join(",") !== expectedLogs.join(",")) {throw new Error("Static OpenMP SDK stage logs are incomplete");}
  const inputs = /** @type {Record<string, string>} */ (Object.create(null));
  /** @param {string} file @param {string} hash */
  const bind = (file, hash) => {
    if (!absolute(file) || !isHash(hash) || (inputs[file] !== undefined && inputs[file] !== hash)) {throw new Error("Conflicting static OpenMP input identity");}
    inputs[file] = hash;
  };
  bind(record.archive.path, record.archive.sha256); bind(record.fetchReceipt.path, record.fetchReceipt.sha256); bind(receiptPath, receiptSha256);
  const notice = verifyOpenmpNoticeInput(root, source.license);
  bind(notice.file, notice.notice.sha256);
  bind(path.join(root, "tools.json"), record.config.sha256); bind(plan.buildArchive, buildOutput.sha256); bind(plan.installArchive, sdkOutput.sha256);
  bind(smokeSourcePath, record.smoke.source.sha256); bind(smokeObjectPath, record.smoke.object.sha256); bind(smokeBinaryPath, record.smoke.binary.sha256);
  for (const row of Object.values(currentTools)) {bind(row.path, row.sha256);}
  for (const entry of Object.values(logs)) {
    if (!entry || typeof entry.path !== "string" || !/^logs\/[a-z-]+\.log$/.test(entry.path) || !isHash(entry.sha256) ||
        !isHash(entry.argvSha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || entry.bytes > logBytesLimit) {
      throw new Error("Invalid static OpenMP stage log receipt");
    }
    const file = path.join(root, entry.path), argv = file + ".argv.json";
    if (regularInput(file, "stage log").size !== entry.bytes || fileHash(file) !== entry.sha256 ||
        regularInput(argv, "stage command receipt").size > (1 << 20) || fileHash(argv) !== entry.argvSha256 ||
        !isDeepStrictEqual(readBuildJson(argv), expectedCommands[path.basename(entry.path, ".log")])) {
      throw new Error("Static OpenMP stage log changed: " + entry.path);
    }
    bind(file, entry.sha256); bind(argv, entry.argvSha256);
  }
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("Static OpenMP SDK input changed: " + file);}}
  if (fileHash(receiptPath) !== receiptSha256) {throw new Error("Static OpenMP SDK receipt changed during admission");}
  return {archive: plan.installArchive, files: inputs, notice: notice.notice, receipt: receiptPath, receiptSha256};
}

if (require.main === module) {
  try {
    const [archive, output, configFile, extra] = process.argv.slice(2);
    if (!archive || !output || !configFile || extra || !absolute(configFile)) {
      throw new Error("Usage: MOM_AOT_BUILD_ONLY=1 node scripts/build-native-openmp-sdk.js /pinned/llvm-project-21.1.8.src.tar.xz /new/openmp-sdk /absolute/tools.json");
    }
    regularInput(configFile, "tool descriptor");
    const config = /** @type {OpenmpConfig} */ (readBuildJson(configFile));
    buildNativeOpenmpSdk(archive, output, config, process.env, configFile, fileHash(configFile));
  } catch (error) {console.error(error instanceof Error ? error.message : "Static OpenMP SDK build failed"); process.exitCode = 1;}
}

module.exports = {archiveName, archiveSha256, toolNames, openmpBuildRecipe, openmpSdkPlan, validateOpenmpArchiveListing, sourceInventory, verifyCMakeCache,
  validateTools, regularInput, verifyOpenmpArchiveInput, verifyStaticOutput, verifyOpenmpDynamicDependencies, assertOpenmpReceiptShape,
  openmpSdkEnvironment, toolRecord, buildNativeOpenmpSdk, verifyOpenmpSourceInventory, verifyOpenmpNoticeInput, nativeOpenmpInputs};
