"use strict";

// A same-run transport for the selected original HIP headers; it is not a second ROCm build.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, sourceInputs, verifySourceInputs, readBuildJson, readBuildText} = require("./build-source-inputs");
const {gitSourceComponents} = require("./fetch-native-sdk-sources");
const {sdkSourceFiles} = require("./prepare-native-sdk-sources");
const {hiprtcHeaders} = require("./build-native-rocm-resources");
const {validCiSuffix} = require("./native-ci-job");
const {canonicalDirectory} = require("./native-ci-tar");

const repo = path.resolve(__dirname, ".."), format = "mom-rocm-hip-headers-handoff-v1",
  patchName = "scripts/patches/rocm-hip-headers-only.patch",
  recipeFiles = ["scripts/native-rocm-hip-headers.js", "scripts/fetch-native-sdk-sources.js", "scripts/prepare-native-sdk-sources.js",
    "scripts/build-native-rocm-resources.js", "scripts/build-native-rocm-host-runtime.js", "scripts/native-comgr-inputs.js", "scripts/build-source-inputs.js",
    "scripts/native-ci-tar.js", patchName],
  maxManifestBytes = 8 << 20, maxFileBytes = 8 << 20, maxFiles = 4096, maxTotalBytes = 64 * 1024 ** 2,
  maxDirectories = 4096, maxDepth = 16;

/** @param {unknown} value @returns {value is string} */
function digest(value) {return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);}

/** @param {unknown} value @returns {value is string} */
function safeRelative(value) {
  return typeof value === "string" && value.length <= 1024 && !value.startsWith("/") && !value.includes("\\") && !/[<>:"|?*]/.test(value) &&
    !Array.from(value).some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127) &&
    value.split("/").every(part => !!part && part !== "." && part !== ".." && !part.endsWith(" ") && !part.endsWith("."));
}

/** @param {string} file @param {number} maximum @param {string} label */
function stableRegular(file, maximum, label) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n || before.size < 1n || before.size > BigInt(maximum) ||
      fs.realpathSync(file) !== file) {throw new Error(label + " must be a bounded canonical singly-linked file");}
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || before.mode !== after.mode) {
    throw new Error(label + " changed while it was hashed");
  }
  return {bytes: Number(after.size), sha256, identity: {dev: before.dev, ino: before.ino, size: before.size,
    mtimeNs: before.mtimeNs, ctimeNs: before.ctimeNs, mode: before.mode}};
}

/** @param {string} root @param {string} label */
function inventoryInclude(root, label) {
  canonicalDirectory(root, label);
  /** @type {Array<{path:string,bytes:number,sha256:string,identity:ReturnType<typeof stableRegular>["identity"]}>} */ const files = [];
  const folded = new Set(); let bytes = 0, directories = 0, entries = 0;
  /** @param {string} directory @param {string} relative @param {number} depth */
  function visit(directory, relative, depth) {
    if (depth > maxDepth) {throw new Error(label + " exceeds its directory-depth bound");}
    const names = fs.readdirSync(directory).sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
    for (const name of names) {
      if (!name || /[\\/:\r\n\0]/.test(name) || Array.from(name).some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127) ||
          ++entries > maxFiles + maxDirectories) {throw new Error(label + " contains an unsafe or excessive directory entry");}
      const file = path.join(directory, name), member = relative ? relative + "/" + name : name,
        stat = fs.lstatSync(file);
      if (!safeRelative(member) || stat.isSymbolicLink()) {throw new Error(label + " contains an unsafe path or symbolic link");}
      const key = member.normalize("NFC").toLowerCase();
      if (folded.has(key)) {throw new Error(label + " contains a case or Unicode path alias");}
      folded.add(key);
      if (stat.isDirectory()) {
        if (++directories > maxDirectories) {throw new Error(label + " exceeds its directory-count bound");}
        visit(file, member, depth + 1); continue;
      }
      if (!stat.isFile() || files.length >= maxFiles) {throw new Error(label + " contains a special file or too many headers");}
      const pin = stableRegular(file, maxFileBytes, label + " member " + member);
      bytes += pin.bytes;
      if (!Number.isSafeInteger(bytes) || bytes > maxTotalBytes) {throw new Error(label + " exceeds its total-byte bound");}
      files.push({path: member, bytes: pin.bytes, sha256: pin.sha256, identity: pin.identity});
    }
  }
  visit(root, "", 0);
  if (!files.length) {throw new Error(label + " is empty");}
  return files.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
}

/** @param {unknown} value @param {string} label */
function requireSourceSummary(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("ROCm HIP header package has invalid " + label + " source-export evidence");
  }
  const row = /** @type {Record<string, any>} */ (value);
  if (!hasExactBuildFields(row, ["fileCount", "filesSha256", "revision", "root", "treeSha256"]) ||
      typeof row["root"] !== "string" || !Number.isSafeInteger(row["fileCount"]) || row["fileCount"] < 1 || row["fileCount"] > 100000 ||
      !digest(row["filesSha256"]) || !digest(row["treeSha256"]) || !/^[a-f0-9]{40}$/.test(row["revision"] || "")) {
    throw new Error("ROCm HIP header package has invalid " + label + " source-export evidence");
  }
}

/** Check only fields that are carried with the selected headers; the original public producer
 * already validated the complete Git exports and configure inputs before writing this package.
 * @param {unknown} input @param {string} sourceRoot @param {boolean} verifyOriginal */
function validateHeaderPackage(input, sourceRoot, verifyOriginal) {
  const packageRow = /** @type {any} */ (input), pins = gitSourceComponents("rocm"), hip = pins["hip"]?.revision, clr = pins["clr"]?.revision;
  const suffix = typeof sourceRoot === "string" ? path.basename(sourceRoot).slice("mom-native-rocm-hip-headers-".length) : "";
  if (!validCiSuffix(suffix) || sourceRoot !== path.join("/tmp", "mom-native-rocm-hip-headers-" + suffix)) {
    throw new Error("ROCm HIP header package source root is not the canonical run path");
  }
  const sourceCohortRoot = path.join("/tmp", "mom-native-rocm-prerequisites-" + suffix);
  if (!packageRow || !hasExactBuildFields(packageRow, ["version", "pins", "sourceExports", "patch", "configure", "hiprtcCmake",
    "generatedHeader", "files", "requiredHiprtcHeaders", "buildOnly", "publicationQualified"]) ||
      packageRow.version !== 1 || packageRow.buildOnly !== true || packageRow.publicationQualified !== false ||
      !hasExactBuildFields(packageRow.pins, ["hip", "clr"]) || packageRow.pins.hip !== hip || packageRow.pins.clr !== clr ||
      !hasExactBuildFields(packageRow.sourceExports, ["hipCommon", "clrHipamd"])) {
    throw new Error("ROCm HIP header package is not the current pinned build-only producer output");
  }
  requireSourceSummary(packageRow.sourceExports.hipCommon, "HIP common");
  const common = packageRow.sourceExports.hipCommon, clrExport = packageRow.sourceExports.clrHipamd;
  if (common.root !== "source/hip-common" || common.revision !== hip || !clrExport || !hasExactBuildFields(clrExport,
    ["afterPatch", "fileCount", "filesSha256", "revision", "root", "treeish", "treeSha256"]) ||
      clrExport.root !== "source/clr-hipamd" || clrExport.revision !== clr || clrExport.treeish !== clr + ":hipamd" ||
      !Number.isSafeInteger(clrExport.fileCount) || clrExport.fileCount < 1 || clrExport.fileCount > 100000 ||
      !digest(clrExport.filesSha256) || !digest(clrExport.treeSha256) || !hasExactBuildFields(clrExport.afterPatch, ["fileCount", "filesSha256"]) ||
      clrExport.afterPatch.fileCount !== clrExport.fileCount || !digest(clrExport.afterPatch.filesSha256)) {
    throw new Error("ROCm HIP header package source exports do not match the current HIP/CLR pins");
  }
  const patch = packageRow.patch, patchFile = path.join(repo, patchName);
  if (!hasExactBuildFields(patch, ["path", "sha256", "stripComponents", "target", "originalSha256", "patchedSha256", "checkLog", "applyLog"]) ||
      patch.path !== "patches/rocm-hip-headers-only.patch" || patch.sha256 !== fileHash(patchFile) || patch.stripComponents !== 2 ||
      patch.target !== "source/clr-hipamd/CMakeLists.txt" || !digest(patch.originalSha256) || !digest(patch.patchedSha256) ||
      patch.checkLog !== "patches/check.json" || patch.applyLog !== "patches/apply.json") {
    throw new Error("ROCm HIP header package patch differs from the current reviewed source patch");
  }
  const configure = packageRow.configure;
  if (!hasExactBuildFields(configure, ["command", "environment", "configureOnly", "log", "logSha256", "status"]) ||
      configure.configureOnly !== true || configure.status !== 0 || configure.log !== "cmake-configure.log" || !digest(configure.logSha256) ||
      !hasExactBuildFields(configure.environment, ["GIT_DIR"]) || configure.environment.GIT_DIR !== path.join(sourceCohortRoot, "roots/clr") ||
      !Array.isArray(configure.command) || configure.command.length < 12 || configure.command.length > 64 ||
      configure.command.some((/** @type {unknown} */ item) => typeof item !== "string" || item.length > 4096 || /[\r\n\0]/.test(item))) {
    throw new Error("ROCm HIP header package lacks a successful configure-only receipt");
  }
  const command = /** @type {string[]} */ (configure.command), prefix = path.join(sourceRoot, "build/install-prefix"),
    expectArgs = ["-S", path.join(sourceRoot, "source/clr-hipamd"), "-B", path.join(sourceRoot, "build"), "-G", "Ninja",
      "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=" + prefix,
      "-DHIP_COMMON_DIR=" + path.join(sourceRoot, "source/hip-common"), "-DHIP_PLATFORM:STRING=amd", "-DHIP_RUNTIME:STRING=none",
      "-DHIPCC_BIN_DIR:STRING=", "-DHIP_OFFICIAL_BUILD:BOOL=ON", "-DROCM_PATH=" + prefix];
  const cCompiler = command.filter(item => item.startsWith("-DCMAKE_C_COMPILER=")), cxxCompiler = command.filter(item => item.startsWith("-DCMAKE_CXX_COMPILER=")),
    cCompilerPath = cCompiler[0]?.slice("-DCMAKE_C_COMPILER=".length) || "", cxxCompilerPath = cxxCompiler[0]?.slice("-DCMAKE_CXX_COMPILER=".length) || "";
  const expectedCommand = [command[0], ...expectArgs.slice(0, 8), "-DCMAKE_C_COMPILER=" + cCompilerPath,
    "-DCMAKE_CXX_COMPILER=" + cxxCompilerPath, ...expectArgs.slice(8)];
  if (!path.isAbsolute(command[0] || "") || cCompiler.length !== 1 || !path.isAbsolute(cCompilerPath) ||
      cxxCompiler.length !== 1 || !path.isAbsolute(cxxCompilerPath) || !isDeepStrictEqual(command, expectedCommand)) {
    throw new Error("ROCm HIP header configure command differs from the headers-only producer plan");
  }
  const hiprtc = packageRow.hiprtcCmake, generated = packageRow.generatedHeader;
  if (!hasExactBuildFields(hiprtc, ["path", "sha256"]) || hiprtc.path !== "source/clr-hipamd/src/hiprtc/CMakeLists.txt" || !digest(hiprtc.sha256) ||
      !hasExactBuildFields(generated, ["path", "sha256"]) || generated.path !== "build/include/hip/hip_version.h" || !digest(generated.sha256) ||
      !packageRow.files || typeof packageRow.files !== "object" || Array.isArray(packageRow.files) ||
      Object.keys(packageRow.files).length < 17 || Object.keys(packageRow.files).length > maxFiles ||
      !Array.isArray(packageRow.requiredHiprtcHeaders) || packageRow.requiredHiprtcHeaders.length < 16 || packageRow.requiredHiprtcHeaders.length > 128) {
    throw new Error("ROCm HIP header package has an incomplete bounded header inventory");
  }
  const names = Object.keys(packageRow.files).sort();
  if (names.some(name => !safeRelative(name) || !digest(packageRow.files[name])) ||
      new Set(names.map(name => name.normalize("NFC").toLowerCase())).size !== names.length ||
      !["hip/hip_runtime_api.h", "hip/hiprtc.h", "hip/hip_version.h"].every(name => names.includes(name)) ||
      packageRow.files["hip/hip_version.h"] !== generated.sha256) {
    throw new Error("ROCm HIP header package file map is invalid or misses a required public header");
  }
  const required = packageRow.requiredHiprtcHeaders;
  if (required.some((/** @type {unknown} */ name) => !safeRelative(name) ||
      !/^hip\/[A-Za-z0-9_./-]+\.h$/.test(name) || !Object.hasOwn(packageRow.files, name))) {
    throw new Error("ROCm HIPRTC header list is unsafe or not covered by the package map");
  }
  if (verifyOriginal) {
    const sourceExportsRoot = path.join(sourceRoot, "source"), commonRoot = path.join(sourceExportsRoot, "hip-common"),
      clrRoot = path.join(sourceExportsRoot, "clr-hipamd"), gitRoot = path.join(sourceCohortRoot, "roots/clr"),
      generatedFile = path.join(sourceRoot, generated.path), hiprtcFile = path.join(sourceRoot, hiprtc.path),
      patchCopy = path.join(sourceRoot, patch.path), configureLog = path.join(sourceRoot, configure.log),
      patchedTarget = path.join(sourceRoot, patch.target);
    const sourceDirectories = /** @type {Array<[string, string]>} */ ([[commonRoot, "HIP common source export"],
      [clrRoot, "CLR HIP AMD source export"], [gitRoot, "original CLR Git root"]]);
    for (const [directory, label] of sourceDirectories) {canonicalDirectory(directory, label);}
    const commonFiles = sdkSourceFiles(commonRoot), clrFiles = sdkSourceFiles(clrRoot),
      mapDigest = (/** @type {Record<string, string>} */ files) => createHash("sha256").update(JSON.stringify(files)).digest("hex");
    if (Object.keys(commonFiles).length !== common.fileCount || mapDigest(commonFiles) !== common.filesSha256 ||
        Object.keys(clrFiles).length !== clrExport.afterPatch.fileCount || mapDigest(clrFiles) !== clrExport.afterPatch.filesSha256 ||
        clrFiles["src/hiprtc/CMakeLists.txt"] !== hiprtc.sha256 || clrFiles["CMakeLists.txt"] !== patch.patchedSha256) {
      throw new Error("ROCm HIP header package no longer matches its original HIP/CLR source exports");
    }
    for (const [name, sha] of Object.entries(packageRow.files)) {
      const sourceName = "include/" + name, sourceHash = name === "hip/hip_version.h" ? generated.sha256 :
        name.startsWith("hip/amd_detail/") ? clrFiles[sourceName] : commonFiles[sourceName];
      if (sourceHash !== sha) {throw new Error("Selected HIP header differs from its original source export: " + name);}
    }
    for (const [file, expected, label] of [[generatedFile, generated.sha256, "Generated HIP version header"],
      [hiprtcFile, hiprtc.sha256, "Original HIPRTC CMake source"], [patchCopy, patch.sha256, "Saved HIP header patch"],
      [configureLog, configure.logSha256, "HIP header configure log"], [patchedTarget, patch.patchedSha256, "Patched CLR HIP CMake file"]]) {
      if (stableRegular(file, maxManifestBytes, label).sha256 !== expected) {throw new Error(label + " differs from its producer receipt");}
    }
    const actualRequired = hiprtcHeaders(readBuildText(hiprtcFile, 1 << 20), path.join(sourceRoot, "include"))
      .map(file => path.relative(path.join(sourceRoot, "include"), file).split(path.sep).join("/"));
    if (actualRequired.join("\0") !== required.join("\0")) {throw new Error("ROCm HIPRTC header list differs from the original CMake source");}
    for (const logName of [patch.checkLog, patch.applyLog]) {
      const logFile = path.join(sourceRoot, logName);
      stableRegular(logFile, maxManifestBytes, "HIP header patch log");
      const evidence = readBuildJson(logFile);
      const check = logName === patch.checkLog, expectedArgs = ["-C", path.join(sourceRoot, "source/clr-hipamd"), "apply", "-p2",
        ...(check ? ["--check"] : []), "--", patchCopy];
      if (!hasExactBuildFields(evidence, ["command", "status", "signal", "error", "stdout", "stderr"]) || evidence.status !== 0 ||
          evidence.signal !== null || evidence.error !== null || typeof evidence.stdout !== "string" || typeof evidence.stderr !== "string" ||
          !Array.isArray(evidence.command) || evidence.command.length !== expectedArgs.length + 1 || !path.isAbsolute(evidence.command[0] || "") ||
          !isDeepStrictEqual(evidence.command.slice(1), expectedArgs)) {
        throw new Error("ROCm HIP header patch receipt did not pass: " + path.basename(logName));
      }
    }
  }
  return packageRow;
}

/** Export only the unchanged HIP header package and its original package receipt.
 * @param {string} sourceRoot @param {string} newOutputRoot @param {string} suffix CI work-root suffix of both roots */
function exportRocmHipHeaders(sourceRoot, newOutputRoot, suffix) {
  if (!validCiSuffix(suffix)) {throw new Error("ROCm HIP header export needs the CI work-root suffix");}
  canonicalDirectory(sourceRoot, "Original ROCm HIP header package root");
  const expectedSource = path.join("/tmp", "mom-native-rocm-hip-headers-" + suffix), outputParent = path.dirname(newOutputRoot);
  if (sourceRoot !== expectedSource || !path.isAbsolute(newOutputRoot) || path.resolve(newOutputRoot) !== newOutputRoot ||
      path.basename(newOutputRoot) !== "native-rocm-hip-headers-" + suffix || fs.existsSync(newOutputRoot)) {
    throw new Error("ROCm HIP header handoff paths differ from the CI work roots");
  }
  canonicalDirectory(outputParent, "ROCm HIP header handoff parent");
  const recipe = sourceInputs(repo, [], recipeFiles), packagePath = path.join(sourceRoot, "header-package.json"),
    packagePin = stableRegular(packagePath, maxManifestBytes, "Original HIP header package receipt"),
    packageRow = validateHeaderPackage(readBuildJson(packagePath), sourceRoot, true),
    includeSource = path.join(sourceRoot, "include"), before = inventoryInclude(includeSource, "Original HIP header include tree"),
    sourceMap = Object.fromEntries(before.map(row => [row.path, row.sha256]));
  if (!isExactMap(sourceMap, packageRow.files)) {throw new Error("Original HIP headers differ from their file receipt");}

  const sourceOverlapsOutput = path.relative(sourceRoot, newOutputRoot), outputOverlapsSource = path.relative(newOutputRoot, sourceRoot);
  if (inside(sourceOverlapsOutput) || inside(outputOverlapsSource)) {throw new Error("ROCm HIP header export must not overlap its source");}
  fs.mkdirSync(newOutputRoot, {mode: 0o755});
  const includeOutput = path.join(newOutputRoot, "include"); fs.mkdirSync(includeOutput, {mode: 0o755});
  for (const row of before) {
    const source = path.join(includeSource, ...row.path.split("/")), destination = path.join(includeOutput, ...row.path.split("/"));
    fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o755});
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
    if (stableRegular(destination, maxFileBytes, "Copied HIP header " + row.path).sha256 !== row.sha256 ||
        stableRegular(source, maxFileBytes, "Source HIP header " + row.path).identity.ino !== row.identity.ino) {
      throw new Error("Copied HIP header changed during same-run export: " + row.path);
    }
  }
  const packageOutput = path.join(newOutputRoot, "header-package.json");
  fs.copyFileSync(packagePath, packageOutput, fs.constants.COPYFILE_EXCL);
  if (stableRegular(packageOutput, maxManifestBytes, "Copied HIP header package receipt").sha256 !== packagePin.sha256) {
    throw new Error("Copied HIP header receipt differs from original bytes");
  }
  const after = inventoryInclude(includeSource, "Original HIP header include tree"), copied = inventoryInclude(includeOutput, "Copied HIP header include tree"),
    afterMap = Object.fromEntries(after.map(row => [row.path, row.sha256]));
  if (!isExactMap(afterMap, packageRow.files) || !sameInventory(before, after) || !sameInventory(before, copied) ||
      stableRegular(packagePath, maxManifestBytes, "Original HIP header package receipt").sha256 !== packagePin.sha256) {
    throw new Error("ROCm HIP header source changed during export");
  }
  const payload = Object.fromEntries([
    ["header-package.json", {bytes: packagePin.bytes, sha256: packagePin.sha256}],
    ...copied.map(row => ["include/" + row.path, {bytes: row.bytes, sha256: row.sha256}]),
  ]);
  const handoff = {version: 1, format, sourceRoot, manifestSha256: packagePin.sha256, recipe, files: payload},
    handoffPath = path.join(newOutputRoot, "handoff.json");
  fs.writeFileSync(handoffPath, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  const handoffSha256 = stableRegular(handoffPath, maxManifestBytes, "ROCm HIP header handoff receipt").sha256;
  const files = Object.fromEntries([[packageOutput, packagePin.sha256], ...copied.map(row => [path.join(includeOutput, ...row.path.split("/")), row.sha256])]);
  return {root: newOutputRoot, include: includeOutput, headerPackage: packageOutput, handoffPath,
    manifestSha256: packagePin.sha256, handoffSha256, fileCount: copied.length, bytes: copied.reduce((sum, row) => sum + row.bytes, 0), files};
}

/** Admit an externally hash-pinned, same-run HIP header transport after the existing tar layer restores it.
 * The archive pin is enforced by that layer; this reader binds restored files back to the transported manifest.
 * @param {string} root @param {{handoffSha256:string}} expected */
function rocmHipHeaderInputs(root, expected) {
  if (!expected || !digest(expected.handoffSha256)) {throw new Error("ROCm HIP header reader needs its external handoff pin");}
  const suffix = typeof root === "string" ? path.basename(root).slice("native-rocm-hip-headers-".length) : "";
  if (!validCiSuffix(suffix) || root !== path.join("/tmp", "native-rocm-hip-headers-" + suffix)) {throw new Error("Restored ROCm HIP header root differs from the canonical run path");}
  canonicalDirectory(root, "Restored ROCm HIP header root");
  if (!isDeepStrictEqual(fs.readdirSync(root).sort(), ["handoff.json", "header-package.json", "include"])) {
    throw new Error("Restored ROCm HIP header root has missing or extra top-level members");
  }
  const handoffPath = path.join(root, "handoff.json"), handoffPin = stableRegular(handoffPath, maxManifestBytes, "ROCm HIP header handoff receipt");
  if (handoffPin.sha256 !== expected.handoffSha256) {throw new Error("ROCm HIP header handoff differs from its external hash pin");}
  const handoff = readBuildJson(handoffPath);
  if (!hasExactBuildFields(handoff, ["version", "format", "sourceRoot", "manifestSha256", "recipe", "files"]) ||
      handoff.version !== 1 || handoff.format !== format ||
      handoff.sourceRoot !== path.join("/tmp", "mom-native-rocm-hip-headers-" + suffix) || !digest(handoff.manifestSha256) ||
      !handoff.recipe || typeof handoff.recipe !== "object" || !handoff.files || typeof handoff.files !== "object" || Array.isArray(handoff.files)) {
    throw new Error("ROCm HIP header handoff identity or schema is invalid");
  }
  verifySourceInputs(handoff.recipe, sourceInputs(repo, [], recipeFiles), "ROCm HIP header handoff recipe");
  const packagePath = path.join(root, "header-package.json"), packagePin = stableRegular(packagePath, maxManifestBytes, "Restored HIP header package receipt"),
    packageRow = validateHeaderPackage(readBuildJson(packagePath), handoff.sourceRoot, false),
    include = path.join(root, "include"), actual = inventoryInclude(include, "Restored HIP header include tree"),
    expectedMap = Object.fromEntries(actual.map(row => [row.path, row.sha256])), payloadRows = handoff.files;
  if (packagePin.sha256 !== handoff.manifestSha256 || !isExactMap(expectedMap, packageRow.files)) {
    throw new Error("Restored HIP header package differs from its original manifest");
  }
  const expectedPayload = {"header-package.json": {bytes: packagePin.bytes, sha256: packagePin.sha256},
    ...Object.fromEntries(actual.map(row => ["include/" + row.path, {bytes: row.bytes, sha256: row.sha256}]))};
  if (!isDeepExactObject(payloadRows, expectedPayload)) {throw new Error("Restored HIP header payload differs from its externally pinned handoff");}
  const files = Object.fromEntries([[packagePath, packagePin.sha256], ...actual.map(row => [path.join(include, ...row.path.split("/")), row.sha256])]),
    after = inventoryInclude(include, "Restored HIP header include tree");
  if (!sameInventory(actual, after) || stableRegular(packagePath, maxManifestBytes, "Restored HIP header package receipt").sha256 !== packagePin.sha256 ||
      stableRegular(handoffPath, maxManifestBytes, "ROCm HIP header handoff receipt").sha256 !== expected.handoffSha256) {
    throw new Error("ROCm HIP header handoff changed during admission");
  }
  return {root, include, headerPackage: packagePath, sourceRoot: handoff.sourceRoot, files,
    manifestSha256: packagePin.sha256, handoffSha256: handoffPin.sha256,
    fileCount: actual.length, bytes: actual.reduce((sum, row) => sum + row.bytes, 0)};
}

/** @param {Record<string,string>} actual @param {Record<string,string>} expected */
function isExactMap(actual, expected) {
  return !!expected && typeof expected === "object" && !Array.isArray(expected) &&
    hasExactBuildFields(actual, Object.keys(expected)) &&
    Object.keys(actual).every(key => actual[key] === expected[key]);
}

/** @param {string} relative */
function inside(relative) {return relative === "" || !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep);}

/** @param {Array<{path:string,bytes:number,sha256:string}>} left @param {Array<{path:string,bytes:number,sha256:string}>} right */
function sameInventory(left, right) {
  return left.length === right.length && left.every((row, index) => row.path === right[index]?.path && row.bytes === right[index]?.bytes && row.sha256 === right[index]?.sha256);
}

/** @param {unknown} actual @param {unknown} expected */
function isDeepExactObject(actual, expected) {
  return actual && expected && typeof actual === "object" && typeof expected === "object" && !Array.isArray(actual) && !Array.isArray(expected) &&
    isDeepStrictEqual(actual, expected);
}

module.exports = {exportRocmHipHeaders, rocmHipHeaderInputs};
