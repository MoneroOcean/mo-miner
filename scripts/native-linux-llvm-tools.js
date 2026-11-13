"use strict";

// A deliberately small same-run transport for the Linux-built tools needed by the
// Windows LLVM host and GPU-factory builds. This is not a second Linux SDK admission path.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {archiveName, archiveSha256} = require("./native-llvm-sdk");
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm21-sdk");

const repo = path.resolve(__dirname, ".."), format = "mom-linux-llvm-tools-handoff-v1",
  acppFrontendFormat = "mom-linux-llvm-acpp-frontend-handoff-v1";
const receiptNames = ["result.json", "configured.json", "install.json", "build-complete.json"];
/** @typedef {{path:string,type:"file"|"symlink",target?:string,sourcePath?:string,bytes?:number,sha256?:string}} SelectionRow */
/** @type {SelectionRow[]} */
const toolRows = [
  {path: "bin/lld", type: "file"}, {path: "bin/llvm-nm", type: "file"}, {path: "bin/llvm-ar", type: "file"},
  {path: "bin/llvm-readobj", type: "file"}, {path: "bin/llvm-objcopy", type: "file"},
  {path: "bin/llvm-rc", type: "file"}, {path: "bin/llvm-ml", type: "file"}, {path: "bin/llvm-tblgen", type: "file"},
  {path: "bin/clang-tblgen", type: "file"}, {path: "bin/lld-link", type: "symlink", target: "lld"},
  {path: "bin/llvm-lib", type: "symlink", target: "llvm-ar"}, {path: "bin/llvm-ranlib", type: "symlink", target: "llvm-ar"}
];
const recipeFiles = ["scripts/native-linux-llvm-tools.js", "scripts/build-native-llvm21-sdk.js", "scripts/native-llvm-sdk.js",
  "scripts/native-audit.js", "scripts/build-source-inputs.js"];
const maxReceiptBytes = 8 << 20, maxArchiveBytes = 2 * 1024 ** 3, maxEntries = 32,
  maxHandoffBytes = 1 << 30, outputReserveBytes = 4 * 1024 ** 3,
  maxFrontendFiles = 4096, maxFrontendBytes = 256 << 20, maxFrontendManifestBytes = 8 << 20;

/** Production callers use the locked archive; this explicit override is only for tiny fixtures.
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}|undefined} testPolicy */
function sourceArchivePolicy(testPolicy) {
  if (testPolicy === undefined) {return {name: archiveName, sha256: archiveSha256};}
  if (!testPolicy || Object.keys(testPolicy).sort().join(",") !== "testOnlySourceArchiveName,testOnlySourceArchiveSha256" ||
      typeof testPolicy.testOnlySourceArchiveName !== "string" || !/^[A-Za-z0-9_.-]{1,128}$/.test(testPolicy.testOnlySourceArchiveName) ||
      testPolicy.testOnlySourceArchiveName === "." || testPolicy.testOnlySourceArchiveName === ".." ||
      !/^[a-f0-9]{64}$/.test(testPolicy.testOnlySourceArchiveSha256 || "")) {
    throw new Error("Linux LLVM test-only source archive policy is malformed");
  }
  return {name: testPolicy.testOnlySourceArchiveName, sha256: testPolicy.testOnlySourceArchiveSha256};
}

/** @param {string} directory */
function availableBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine Linux LLVM tool handoff output capacity");}
  const bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

/** @param {string} value @param {string} label */
function canonicalDirectory(value, label) {
  if (typeof value !== "string" || !path.isAbsolute(value) || path.resolve(value) !== value || /[\r\n\0]/.test(value)) {
    throw new Error(label + " must be an absolute canonical path");
  }
  const stat = fs.lstatSync(value);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(value) !== value) {throw new Error(label + " must be a real directory");}
  return value;
}

/** @param {string} file @param {number} maximum @param {string} label */
function stableRegular(file, maximum, label) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n || before.size < 1n || before.size > BigInt(maximum) ||
      path.resolve(fs.realpathSync(file)) !== path.resolve(file)) {throw new Error(label + " must be a bounded canonical singly-linked file");}
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || before.mode !== after.mode) {
    throw new Error(label + " changed while it was hashed");
  }
  return {bytes: Number(after.size), sha256, mode: Number(after.mode & 0o777n), identity: {dev: before.dev, ino: before.ino,
    size: before.size, mtimeNs: before.mtimeNs, ctimeNs: before.ctimeNs, mode: before.mode}};
}

/** @param {string} file @param {ReturnType<typeof stableRegular>['identity']} identity @param {string} label */
function sameInput(file, identity, label) {
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1n || stat.dev !== identity.dev || stat.ino !== identity.ino ||
      stat.size !== identity.size || stat.mtimeNs !== identity.mtimeNs || stat.ctimeNs !== identity.ctimeNs || stat.mode !== identity.mode) {
    throw new Error(label + " changed during handoff production");
  }
}

/** @param {string} file @param {string} expectedTarget @param {string} root @param {string} label */
function stableAlias(file, expectedTarget, root, label) {
  const before = fs.lstatSync(file, {bigint: true}), targetText = fs.readlinkSync(file), target = path.resolve(path.dirname(file), targetText),
    relative = path.relative(root, target);
  if (!before.isSymbolicLink() || before.nlink !== 1n || targetText !== expectedTarget || path.isAbsolute(targetText) ||
      path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || fs.realpathSync(file) !== fs.realpathSync(target)) {
    throw new Error(label + " is not the exact safe relative alias");
  }
  return {target: targetText, identity: {dev: before.dev, ino: before.ino, size: before.size, mtimeNs: before.mtimeNs,
    ctimeNs: before.ctimeNs, mode: before.mode}};
}

/** @param {string} file @param {ReturnType<typeof stableAlias>['identity']} identity @param {string} target @param {string} label */
function sameAlias(file, identity, target, label) {
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isSymbolicLink() || stat.dev !== identity.dev || stat.ino !== identity.ino || stat.size !== identity.size ||
      stat.mtimeNs !== identity.mtimeNs || stat.ctimeNs !== identity.ctimeNs || stat.mode !== identity.mode || fs.readlinkSync(file) !== target) {
    throw new Error(label + " changed during handoff production");
  }
}

/** @param {string} file @param {string} destination @param {string} expectedSha256 @param {number} maximum @param {string} label */
function copyPinned(file, destination, expectedSha256, maximum, label) {
  const before = stableRegular(file, maximum, label);
  if (before.sha256 !== expectedSha256) {throw new Error(label + " does not match the admitted receipt");}
  fs.copyFileSync(file, destination, fs.constants.COPYFILE_EXCL);
  fs.chmodSync(destination, before.mode & 0o755);
  const copied = stableRegular(destination, maximum, label + " copy");
  sameInput(file, before.identity, label);
  if (copied.sha256 !== before.sha256 || copied.bytes !== before.bytes || copied.mode !== (before.mode & 0o755)) {
    throw new Error(label + " copy differs from its source");
  }
  return {bytes: copied.bytes, sha256: copied.sha256, mode: copied.mode};
}

/** @param {string} root @param {number} maximum */
function listExactFiles(root, maximum = maxEntries, maxDepth = 2) {
  /** @type {string[]} */
  const found = [];
  let entries = 0;
  /** @param {string} directory @param {number} depth */
  const walk = (directory, depth) => {
    if (depth > maxDepth) {throw new Error("Linux LLVM tool handoff is too deeply nested");}
    for (const name of fs.readdirSync(directory).sort()) {
      if (!name || name === "." || name === ".." || /[\\/\r\n\0]/.test(name) || ++entries > maximum) {
        throw new Error("Linux LLVM tool handoff inventory exceeds its member bound");
      }
      const absolute = path.join(directory, name), relative = path.relative(root, absolute).split(path.sep).join("/"), stat = fs.lstatSync(absolute);
      if (stat.isDirectory() && !stat.isSymbolicLink()) {walk(absolute, depth + 1); continue;}
      if (!stat.isFile() && !stat.isSymbolicLink()) {throw new Error("Linux LLVM tool handoff contains a special file");}
      found.push(relative);
    }
  };
  walk(root, 0);
  return found.sort();
}

/** @param {string} relative */
function safeMemberPath(relative) {
  return typeof relative === "string" && relative.length <= 1024 && !relative.startsWith("/") && !relative.includes("\\") &&
    !relative.split("").some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127) &&
    relative.split("/").every(part => part && part !== "." && part !== "..");
}

/** Select only the LLVM 21 clang drivers and the installed Clang resource include tree.
 * @param {Record<string, any>} snapshot @returns {SelectionRow[]} */
function acppFrontendRows(snapshot) {
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {throw new Error("LLVM admission omitted its frontend snapshot");}
  /** @type {SelectionRow[]} */ const fixed = [
    {path: "bin/clang", type: "symlink", target: "clang-21"},
    {path: "bin/clang++", type: "symlink", target: "clang"},
    {path: "bin/clang-21", type: "file"}
  ];
  for (const spec of fixed) {
    const row = snapshot[spec.path];
    if (!row || row.type !== spec.type || spec.type === "symlink" && row.target !== spec.target ||
        spec.type === "file" && (!Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (512 << 20) || !/^[a-f0-9]{64}$/.test(row.sha256 || ""))) {
      throw new Error("LLVM 21 frontend driver differs from its admitted snapshot: " + spec.path);
    }
  }
  const prefix = "lib/clang/21/include/", selected = Object.keys(snapshot).filter(name => name.startsWith(prefix));
  if (!selected.length || selected.length > maxFrontendFiles) {throw new Error("LLVM 21 Clang resource headers are missing or exceed the member bound");}
  let bytes = 0;
  const resources = /** @type {SelectionRow[]} */ (selected.sort().map(name => {
    const row = snapshot[name];
    if (!safeMemberPath(name) || !row || row.type !== "file" ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (16 << 20) || !/^[a-f0-9]{64}$/.test(row.sha256 || "")) {
      throw new Error("LLVM 21 Clang resource tree contains an unsupported or invalid member: " + name);
    }
    bytes += row.bytes;
    if (!Number.isSafeInteger(bytes) || bytes > maxFrontendBytes) {throw new Error("LLVM 21 Clang resource headers exceed the byte bound");}
    return {path: name, sourcePath: name, type: "file", bytes: row.bytes, sha256: row.sha256};
  }));
  return [...fixed, ...resources];
}

/** @param {string} root @param {string} label */
function readReceipt(root, label) {
  return /** @type {any} */ (readBuildJson(path.join(root, label)));
}

/** @param {string} root @param {Record<string, {bytes:number,sha256:string}>} pins @param {string} expectedArchiveSha256 */
function verifyReceiptLinks(root, pins, expectedArchiveSha256) {
  const result = readReceipt(root, "receipts/result.json"), configured = readReceipt(root, "receipts/configured.json"),
    install = readReceipt(root, "receipts/install.json"), complete = readReceipt(root, "receipts/build-complete.json");
  const expectedAudit = {files: result.sdk?.files, bytes: result.sdk?.bytes, archives: result.archives, snapshot: result.sdk?.snapshot};
  if (result.version !== 1 || result.role !== "linux-llvm21-source-sdk" || result.stage !== "audited" ||
      result.buildOnly !== true || result.compiled !== true || result.executed !== false || result.publicationQualified !== false ||
      result.sdk?.path !== path.join(result.plan?.output || "", "sdk") || result.archiveSha256 !== expectedArchiveSha256 ||
      configured.archiveSha256 !== expectedArchiveSha256 || result.planSha256 !== configured.planSha256 ||
      !isDeepStrictEqual(result.plan, configured.plan) || !isDeepStrictEqual(result.configure, configured.configure) ||
      !isDeepStrictEqual(result.sourceState, configured.sourceState) || !isDeepStrictEqual(result.sourceInputs, configured.inputFiles) ||
      configured.version !== 1 || configured.role !== result.role || configured.stage !== "configured" || configured.runId !== result.runId ||
      install.version !== 1 || install.role !== result.role || install.runId !== result.runId ||
      complete.version !== 1 || complete.role !== result.role || complete.runId !== result.runId ||
      result.buildComplete?.path !== completePath(result) || result.install?.path !== installPath(result) ||
      result.buildComplete.sha256 !== pins["receipts/build-complete.json"]?.sha256 ||
      result.install.sha256 !== pins["receipts/install.json"]?.sha256 || install.buildCompleteSha256 !== pins["receipts/build-complete.json"]?.sha256 ||
      !Array.isArray(complete.phases) || !complete.phases.length || !isDeepStrictEqual(install.audit, expectedAudit)) {
    throw new Error("Linux LLVM handoff receipts have inconsistent original admission cross-references");
  }
  return {result, configured, install, complete};
}

/** @param {any} result */
function completePath(result) {return path.join(result.plan.output, "build-complete.json");}
/** @param {any} result */
function installPath(result) {return path.join(result.plan.output, "install.json");}

/** @param {unknown} handoff @param {string} expectedSha256 @param {{name:string,sha256:string}} archivePolicy @param {"tools"|"acpp-frontend"} role */
function validateHandoff(handoff, expectedSha256, archivePolicy, role) {
  if (!handoff || typeof handoff !== "object" || Array.isArray(handoff) || !/^[a-f0-9]{64}$/.test(expectedSha256 || "")) {
    throw new Error("Linux LLVM tool handoff is malformed or lacks its external hash pin");
  }
  const row = /** @type {any} */ (handoff);
  const expectedKeys = role === "tools" ? "archiveSha256,buildOnly,compiled,files,format,publicationQualified,receipts,recipe,runAttempt,runId,sourceArchive,version" :
      "archiveSha256,buildOnly,compiled,files,format,frontend,publicationQualified,receipts,recipe,runAttempt,runId,sourceArchive,version",
    expectedFormat = role === "tools" ? format : acppFrontendFormat;
  if (Object.keys(row).sort().join(",") !== expectedKeys ||
      row.version !== 1 || row.format !== expectedFormat || typeof row.runAttempt !== "string" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(row.runAttempt) ||
      typeof row.runId !== "string" || !/^[0-9a-f-]{36}$/.test(row.runId) || row.buildOnly !== true || row.compiled !== true ||
      row.publicationQualified !== false || !/^[a-f0-9]{64}$/.test(row.archiveSha256 || "") || !Array.isArray(row.files) ||
      row.files.length < (role === "tools" ? toolRows.length : 3) + receiptNames.length + 1 ||
      row.files.length > (role === "tools" ? toolRows.length : maxFrontendFiles + 3) + receiptNames.length + 1 ||
      !row.receipts || typeof row.receipts !== "object" || Array.isArray(row.receipts) ||
      Object.keys(row.receipts).sort().join(",") !== receiptNames.slice().sort().join(",") || !row.sourceArchive ||
      row.sourceArchive.path !== archivePolicy.name || row.sourceArchive.sha256 !== archivePolicy.sha256 ||
      row.sourceArchive.sha256 !== row.archiveSha256 || !Number.isSafeInteger(row.sourceArchive.bytes) ||
      row.sourceArchive.bytes < 1 || row.sourceArchive.bytes > maxArchiveBytes ||
      !row.recipe || typeof row.recipe !== "object" || Array.isArray(row.recipe)) {
    throw new Error("Linux LLVM tool handoff is invalid or differs from its external hash pin");
  }
  if (role === "acpp-frontend" && (!row.frontend || typeof row.frontend !== "object" || Array.isArray(row.frontend) ||
      Object.keys(row.frontend).sort().join(",") !== "clang,clangxx,resourceInclude,version" || row.frontend.clang !== "bin/clang" ||
      row.frontend.clangxx !== "bin/clang++" || row.frontend.resourceInclude !== "lib/clang/21/include" || row.frontend.version !== "21.1.8")) {
    throw new Error("Linux LLVM ACPP frontend handoff has an unexpected fixed frontend role");
  }
  const listed = new Set();
  for (const item of row.files) {
    if (!item || typeof item !== "object" || typeof item.path !== "string" || listed.has(item.path)) {
      throw new Error("Linux LLVM handoff has a duplicate or malformed payload member");
    }
    listed.add(item.path);
    if (role === "acpp-frontend" && item.path.startsWith("lib/clang/21/include/")) {
      if (!safeMemberPath(item.path) || item.type !== "file") {throw new Error("Linux LLVM frontend resource member is unsafe or unsupported");}
    }
  }
  return row;
}

/** @param {string} bundleRoot @param {{runAttempt:string,handoffSha256:string}} expectation
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}} [testPolicy]
 * @returns {{toolRoot:string,lldLink:string,llvmNm:string,sourceArchive:string,files:Record<string, any>}} */
function linuxLlvmToolHandoffInputs(bundleRoot, expectation, testPolicy) {
  return /** @type {{toolRoot:string,lldLink:string,llvmNm:string,sourceArchive:string,files:Record<string, any>}} */ (
    linuxLlvmHandoffInputs(bundleRoot, expectation, testPolicy, "tools"));
}

/** Admit the separately selected LLVM 21 Clang frontend/resource-header role.
 * @param {string} bundleRoot @param {{runAttempt:string,handoffSha256:string}} expectation
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}} [testPolicy]
 * @returns {{clang:string,clangxx:string,resourceInclude:string,sourceArchive:string,files:Record<string, any>}} */
function linuxLlvmAcppFrontendInputs(bundleRoot, expectation, testPolicy) {
  return /** @type {{clang:string,clangxx:string,resourceInclude:string,sourceArchive:string,files:Record<string, any>}} */ (
    linuxLlvmHandoffInputs(bundleRoot, expectation, testPolicy, "acpp-frontend"));
}

/** @param {string} bundleRoot @param {{runAttempt:string,handoffSha256:string}} expectation
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}|undefined} testPolicy @param {"tools"|"acpp-frontend"} role */
function linuxLlvmHandoffInputs(bundleRoot, expectation, testPolicy, role) {
  const root = canonicalDirectory(bundleRoot, "Linux LLVM tool handoff root");
  const archivePolicy = sourceArchivePolicy(testPolicy);
  if (!expectation || typeof expectation !== "object" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(expectation.runAttempt || "")) {
    throw new Error("Linux LLVM tool handoff needs the exact same-run attempt");
  }
  const handoffPath = path.join(root, "handoff.json"), handoffFile = stableRegular(handoffPath,
    role === "tools" ? 1 << 20 : maxFrontendManifestBytes, "Linux LLVM tool handoff receipt");
  const handoff = readBuildJson(handoffPath), validated = validateHandoff(handoff, expectation.handoffSha256, archivePolicy, role);
  if (validated.runAttempt !== expectation.runAttempt || handoffFile.sha256 !== expectation.handoffSha256) {
    throw new Error("Linux LLVM tool handoff belongs to another run attempt");
  }
  verifySourceInputs(validated.recipe, sourceInputs(repo, [], recipeFiles), "Linux LLVM tool handoff recipe");
  /** @type {Map<string, {path:string,type:string,target?:string}>} */
  const expectedFiles = new Map();
  /** @type {SelectionRow[]} */ const selectedRows = role === "tools" ? toolRows : [
    {path: "bin/clang", type: "symlink", target: "clang-21"},
    {path: "bin/clang++", type: "symlink", target: "clang"},
    {path: "bin/clang-21", type: "file"},
    ...validated.files.filter((/** @type {any} */ item) => item.path.startsWith("lib/clang/21/include/")).map((/** @type {any} */ item) => ({path: item.path, type: "file"}))
  ];
  for (const row of selectedRows) {expectedFiles.set(row.path, row);}
  expectedFiles.set(archivePolicy.name, {path: archivePolicy.name, type: "file"});
  for (const name of receiptNames) {expectedFiles.set("receipts/" + name, {path: "receipts/" + name, type: "file"});}
  const pins = /** @type {Record<string, any>} */ (Object.fromEntries(validated.files.map((/** @type {any} */ item) => {
    if (!item || typeof item !== "object" || typeof item.path !== "string" || !expectedFiles.has(item.path) ||
        !["file", "symlink"].includes(item.type) ||
        item.type === "file" && (Object.keys(item).sort().join(",") !== "bytes,mode,path,sha256,type" ||
          !Number.isSafeInteger(item.bytes) || item.bytes < 1 || !Number.isSafeInteger(item.mode) || item.mode < 0 || item.mode > 0o777 ||
          !/^[a-f0-9]{64}$/.test(item.sha256 || "")) ||
        item.type === "symlink" && (Object.keys(item).sort().join(",") !== "path,target,type" || typeof item.target !== "string")) {
      throw new Error("Linux LLVM handoff lists an unexpected payload path");
    }
    return [item.path, item];
  })));
  const memberLimit = role === "tools" ? maxEntries : maxFrontendFiles + 3 + receiptNames.length + 3,
    maxDepth = role === "tools" ? 2 : 10;
  if (Object.keys(pins).length !== expectedFiles.size || listExactFiles(root, memberLimit, maxDepth).sort().join("\0") !==
      [...expectedFiles.keys(), "handoff.json"].sort().join("\0")) {
    throw new Error("Linux LLVM handoff has missing or extra payload members");
  }
  for (const spec of expectedFiles.values()) {
    const pin = pins[spec.path], file = path.join(root, spec.path);
    if (spec.type === "symlink") {
      if (!pin || pin.type !== "symlink" || pin.target !== spec.target || fs.lstatSync(file).isSymbolicLink() !== true || fs.readlinkSync(file) !== spec.target) {
        throw new Error("Linux LLVM handoff alias differs from the locked relative target");
      }
      continue;
    }
    const maximum = spec.path === archivePolicy.name ? maxArchiveBytes : spec.path.startsWith("receipts/") ? maxReceiptBytes :
        spec.path.startsWith("lib/clang/21/include/") ? 16 << 20 : 512 << 20,
      actual = stableRegular(file, maximum, "Linux LLVM handoff file " + spec.path);
    if (!pin || pin.type !== "file" || pin.bytes !== actual.bytes || pin.sha256 !== actual.sha256 || pin.mode !== actual.mode ||
        spec.path === "bin/clang-21" && (actual.mode & 0o111) === 0) {
      throw new Error("Linux LLVM handoff file differs from its external manifest: " + spec.path);
    }
  }
  const receipts = Object.fromEntries(receiptNames.map(name => [name, pins["receipts/" + name]]));
  for (const name of receiptNames) {
    if (validated.receipts[name]?.path !== "receipts/" + name || validated.receipts[name]?.sha256 !== receipts[name].sha256) {
      throw new Error("Linux LLVM original receipt pin is malformed: " + name);
    }
  }
  const parsed = verifyReceiptLinks(root, pins, archivePolicy.sha256);
  if (parsed.result.runId !== validated.runId || parsed.result.archiveSha256 !== validated.archiveSha256) {
    throw new Error("Linux LLVM handoff identity differs from its admitted original receipts");
  }
  const admittedTools = parsed.result.sdk.snapshot?.files;
  if (!admittedTools || typeof admittedTools !== "object" || Array.isArray(admittedTools)) {
    throw new Error("Linux LLVM admission receipt omitted the selected tool snapshot");
  }
  /** @type {SelectionRow[]} */ const receiptRows = role === "tools" ? toolRows : acppFrontendRows(admittedTools);
  if (receiptRows.map(row => row.path).sort().join("\0") !== selectedRows.map(row => row.path).sort().join("\0")) {
    throw new Error("Linux LLVM ACPP frontend selection differs from its original admitted snapshot");
  }
  for (const selected of receiptRows) {
    const sourcePath = selected.sourcePath || selected.path,
      pin = pins[selected.path], admitted = admittedTools[sourcePath];
    if (!admitted || admitted.type !== selected.type ||
        selected.type === "file" && (admitted.bytes !== pin.bytes || admitted.sha256 !== pin.sha256) ||
        selected.type === "symlink" && (admitted.target !== selected.target || pin.target !== selected.target)) {
      throw new Error("Linux LLVM handoff member differs from the original admitted SDK snapshot: " + sourcePath);
    }
  }
  const sourceArchive = path.join(root, archivePolicy.name), archive = stableRegular(sourceArchive, maxArchiveBytes, "Locked LLVM source archive");
  if (archive.sha256 !== archivePolicy.sha256 || archive.bytes !== validated.sourceArchive.bytes) {throw new Error("Linux LLVM locked source archive bytes differ");}
  const recipe = sourceInputs(repo, [], recipeFiles);
  verifySourceInputs(validated.recipe, recipe, "Linux LLVM tool handoff recipe");
  if (stableRegular(handoffPath, role === "tools" ? 1 << 20 : maxFrontendManifestBytes, "Linux LLVM tool handoff receipt").sha256 !== expectation.handoffSha256 ||
      listExactFiles(root, memberLimit, maxDepth).join("\0") !== [...expectedFiles.keys(), "handoff.json"].sort().join("\0")) {
    throw new Error("Linux LLVM tool handoff changed during admission");
  }
  const files = Object.fromEntries(validated.files.map((/** @type {any} */ item) => [item.path, item]));
  return role === "tools" ? {toolRoot: path.join(root, "bin"), lldLink: path.join(root, "bin/lld-link"), llvmNm: path.join(root, "bin/llvm-nm"),
    sourceArchive, files} : {clang: path.join(root, "bin/clang"), clangxx: path.join(root, "bin/clang++"),
    resourceInclude: path.join(root, "lib/clang/21/include"), sourceArchive, files};
}

/** Export only the locked tool closure after full public Linux SDK admission.
 * @param {string} admittedSdkRoot @param {string} outputRoot @param {{runAttempt:string}} identity
 * @param {(root:string)=>Record<string, any>} [reader] Internal fixture seam; public CLI uses the full verifier.
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}} [testPolicy] Tiny-fixture-only source archive pin. */
function exportLinuxLlvmTools(admittedSdkRoot, outputRoot, identity, reader = verifyLinuxLlvmAdmission, testPolicy) {
  return exportLinuxLlvmRole(admittedSdkRoot, outputRoot, identity, reader, testPolicy, "tools");
}

/** Export the LLVM 21 Clang frontend and only its resource headers for the ACPP plugin build.
 * @param {string} admittedSdkRoot @param {string} outputRoot @param {{runAttempt:string}} identity
 * @param {(root:string)=>Record<string, any>} [reader] Internal fixture seam; public callers use the full verifier.
 * @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}} [testPolicy] */
function exportLinuxLlvmAcppFrontend(admittedSdkRoot, outputRoot, identity, reader = verifyLinuxLlvmAdmission, testPolicy) {
  return exportLinuxLlvmRole(admittedSdkRoot, outputRoot, identity, reader, testPolicy, "acpp-frontend");
}

/** @param {string} admittedSdkRoot @param {string} outputRoot @param {{runAttempt:string}} identity
 * @param {(root:string)=>Record<string, any>} reader @param {{testOnlySourceArchiveName:string,testOnlySourceArchiveSha256:string}|undefined} testPolicy
 * @param {"tools"|"acpp-frontend"} role */
function exportLinuxLlvmRole(admittedSdkRoot, outputRoot, identity, reader, testPolicy, role) {
  const sdk = canonicalDirectory(admittedSdkRoot, "Admitted Linux LLVM SDK root");
  const archivePolicy = sourceArchivePolicy(testPolicy);
  if (!identity || typeof identity.runAttempt !== "string" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(identity.runAttempt)) {
    throw new Error("Linux LLVM tool export needs the exact same-run attempt");
  }
  const admitted = reader(sdk), result = /** @type {any} */ (admitted);
  if (!result || result.role !== "linux-llvm21-source-sdk" || result.stage !== "audited" || result.buildOnly !== true ||
      result.compiled !== true || result.executed !== false || result.publicationQualified !== false || result.sdk?.path !== path.join(sdk, "sdk") ||
      result.archiveSha256 !== archivePolicy.sha256) {throw new Error("Linux LLVM tool export requires the full public SDK admission receipt");}
  const outputParent = path.dirname(outputRoot);
  canonicalDirectory(outputParent, "Linux LLVM tool handoff parent");
  if (!path.isAbsolute(outputRoot) || path.resolve(outputRoot) !== outputRoot || fs.existsSync(outputRoot)) {
    throw new Error("Linux LLVM tool handoff requires a fresh canonical output root");
  }
  const outputRelative = path.relative(sdk, outputRoot), sourceRelative = path.relative(outputRoot, sdk),
    outputWithinSource = outputRelative === "" || !path.isAbsolute(outputRelative) && outputRelative !== ".." && !outputRelative.startsWith(".." + path.sep),
    sourceWithinOutput = sourceRelative === "" || !path.isAbsolute(sourceRelative) && sourceRelative !== ".." && !sourceRelative.startsWith(".." + path.sep);
  if (outputWithinSource || sourceWithinOutput) {throw new Error("Linux LLVM tool handoff output must be separate from its admitted source SDK");}
  const sdkRoot = path.join(sdk, "sdk"), sourceRoot = sdk;
  if (result.plan?.output !== sourceRoot || result.plan?.prefix !== sdkRoot || fs.realpathSync(sdkRoot) !== sdkRoot) {
    throw new Error("Linux LLVM admission root does not match its recorded SDK paths");
  }
  const recipe = sourceInputs(repo, [], recipeFiles), snapshot = result.sdk.snapshot?.files;
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {throw new Error("Linux LLVM admission omitted its SDK inventory");}
  /** @type {SelectionRow[]} */ const selectedRows = role === "tools" ? toolRows : acppFrontendRows(snapshot);
  if (role === "acpp-frontend") {
    const includeRoot = path.join(sdkRoot, "lib/clang/21/include"), stat = fs.lstatSync(includeRoot);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(includeRoot) !== includeRoot) {
      throw new Error("LLVM 21 Clang resource include root is not a canonical directory");
    }
    const expectedResources = selectedRows.filter(row => typeof row.sourcePath === "string")
        .map(row => /** @type {string} */ (row.sourcePath).slice("lib/clang/21/include/".length)).sort(),
      actualResources = listExactFiles(includeRoot, maxFrontendFiles, 10);
    if (expectedResources.join("\0") !== actualResources.join("\0")) {
      throw new Error("LLVM 21 Clang resource include tree differs from its admitted snapshot");
    }
  }
  /** @type {Map<string, {file:string,row?:ReturnType<typeof stableRegular>,target?:string,identity?:ReturnType<typeof stableAlias>["identity"]}>} */
  const sources = new Map();
  for (const spec of selectedRows) {
    const inputRelative = spec.sourcePath || spec.path, input = path.join(sdkRoot, inputRelative), row = snapshot[inputRelative];
    if (!row || row.type !== spec.type || (spec.type === "symlink" && row.target !== spec.target)) {
      throw new Error("Linux LLVM admitted tool inventory differs from its explicit handoff selector: " + inputRelative);
    }
    if (spec.type === "file") {
      const maximum = inputRelative.startsWith("lib/clang/21/include/") ? 16 << 20 : 512 << 20,
        pin = stableRegular(input, maximum, "Admitted LLVM tool " + inputRelative);
      if (pin.bytes !== row.bytes || pin.sha256 !== row.sha256 || spec.path === "bin/clang-21" && (pin.mode & 0o111) === 0) {
        throw new Error("Admitted LLVM tool bytes or executable mode differ from the public SDK snapshot");
      }
      sources.set(spec.path, {file: input, row: pin});
    } else {
      if (!spec.target) {throw new Error("Selected Linux LLVM alias lacks a locked target");}
      const alias = path.join(sdkRoot, spec.path);
      const target = path.join(sdkRoot, "bin", spec.target);
      if (!fs.lstatSync(alias).isSymbolicLink() || fs.readlinkSync(alias) !== spec.target || fs.realpathSync(alias) !== fs.realpathSync(target)) {
        throw new Error("Admitted LLVM tool alias changed");
      }
      sources.set(spec.path, {file: alias, ...stableAlias(alias, spec.target, sdkRoot, "Admitted LLVM tool alias " + spec.path)});
    }
  }
  const archivePath = path.join(sourceRoot, archivePolicy.name), archivePin = stableRegular(archivePath, maxArchiveBytes, "Admitted LLVM source archive");
  if (archivePin.sha256 !== archivePolicy.sha256) {throw new Error("Admitted LLVM source archive differs from its locked digest");}
  const inputReceipts = /** @type {Record<string, ReturnType<typeof stableRegular> & {file:string}>} */ (Object.fromEntries(receiptNames.map(name => {
    const file = path.join(sourceRoot, name), maximum = maxReceiptBytes;
    return [name, {file, ...stableRegular(file, maximum, "Admitted LLVM receipt " + name)}];
  })));
  const before = Object.fromEntries(Object.entries(inputReceipts).map(([name, value]) => [name, value.sha256]));
  const totalBytes = [...sources.values()].reduce((sum, source) => sum + (source.row?.bytes || Buffer.byteLength(source.target || "")), archivePin.bytes) +
    Object.values(inputReceipts).reduce((sum, receipt) => sum + receipt.bytes, 0);
  const requiredBytes = 2 * totalBytes + outputReserveBytes;
  if (!Number.isSafeInteger(totalBytes) || totalBytes < 1 || totalBytes > maxHandoffBytes || !Number.isSafeInteger(requiredBytes) ||
      availableBytes(outputParent) < requiredBytes) {
    throw new Error("Linux LLVM tool handoff payload exceeds its bound or lacks output/pack space reserve");
  }
  fs.mkdirSync(outputRoot, {mode: 0o700});
  for (const spec of selectedRows) {fs.mkdirSync(path.dirname(path.join(outputRoot, spec.path)), {recursive: true, mode: 0o700});}
  fs.mkdirSync(path.join(outputRoot, "receipts"), {mode: 0o700});
  /** @type {Array<{path:string,type:"file",bytes:number,sha256:string,mode:number}|{path:string,type:"symlink",target:string}>} */
  const files = [];
  for (const spec of selectedRows) {
    const source = sources.get(spec.path), destination = path.join(outputRoot, spec.path);
    if (!source) {throw new Error("Selected Linux LLVM tool disappeared from the admitted snapshot");}
    if (spec.type === "file") {
      if (!source.row) {throw new Error("Selected Linux LLVM executable has no regular-file pin");}
      const maximum = (spec.sourcePath || spec.path).startsWith("lib/clang/21/include/") ? 16 << 20 : 512 << 20,
        copied = copyPinned(source.file, destination, source.row.sha256, maximum, "LLVM tool " + spec.path);
      if (copied.bytes !== source.row.bytes) {throw new Error("LLVM tool changed size while exporting");}
      files.push({path: spec.path, type: "file", ...copied});
    } else {
      if (!source.target) {throw new Error("Selected Linux LLVM alias has no pinned target");}
      fs.symlinkSync(source.target, destination);
      if (fs.readlinkSync(destination) !== source.target) {throw new Error("LLVM tool alias changed during export");}
      files.push({path: spec.path, type: "symlink", target: source.target});
    }
  }
  const archiveCopy = path.join(outputRoot, archivePolicy.name), archive = copyPinned(archivePath, archiveCopy, archivePolicy.sha256, maxArchiveBytes, "Locked LLVM source archive");
  files.push({path: archivePolicy.name, type: "file", ...archive});
  for (const name of receiptNames) {
    const input = inputReceipts[name];
    if (!input) {throw new Error("Admitted LLVM handoff receipt disappeared");}
    const receiptCopy = path.join(outputRoot, "receipts", name), copied = copyPinned(input.file, receiptCopy,
      input.sha256, maxReceiptBytes, "Original LLVM receipt " + name);
    files.push({path: "receipts/" + name, type: "file", ...copied});
  }
  for (const [name, source] of sources) {
    if (source.row) {sameInput(source.file, source.row.identity, "LLVM tool " + name);} else {
      if (!source.identity || !source.target) {throw new Error("Selected Linux LLVM alias lacks a stable input pin");}
      sameAlias(source.file, source.identity, source.target, "LLVM tool alias " + name);
    }
  }
  sameInput(archivePath, archivePin.identity, "Admitted LLVM source archive");
  for (const [name, input] of Object.entries(inputReceipts)) {sameInput(input.file, input.identity, "Admitted LLVM receipt " + name);}
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeFiles), "Linux LLVM tool handoff recipe");
  const handoff = {version: 1, format: role === "tools" ? format : acppFrontendFormat, runAttempt: identity.runAttempt, runId: result.runId, buildOnly: true, compiled: true,
    publicationQualified: false, archiveSha256: archivePolicy.sha256,
    sourceArchive: {path: archivePolicy.name, bytes: archive.bytes, sha256: archive.sha256},
    receipts: Object.fromEntries(receiptNames.map(name => [name, {path: "receipts/" + name, sha256: before[name]}])),
    files: files.sort((a, b) => a.path.localeCompare(b.path)), recipe,
    ...(role === "acpp-frontend" ? {frontend: {clang: "bin/clang", clangxx: "bin/clang++", resourceInclude: "lib/clang/21/include", version: "21.1.8"}} : {})};
  const handoffPath = path.join(outputRoot, "handoff.json");
  const serialized = JSON.stringify(handoff, null, 2) + "\n";
  if (Buffer.byteLength(serialized) > (role === "tools" ? 1 << 20 : maxFrontendManifestBytes)) {throw new Error("Linux LLVM handoff manifest exceeds its bound");}
  fs.writeFileSync(handoffPath, serialized, {flag: "wx", mode: 0o444});
  const handoffSha256 = fileHash(handoffPath);
  return {handoffPath, handoffSha256, fileCount: files.length};
}

module.exports = {exportLinuxLlvmTools, linuxLlvmToolHandoffInputs, exportLinuxLlvmAcppFrontend, linuxLlvmAcppFrontendInputs};
