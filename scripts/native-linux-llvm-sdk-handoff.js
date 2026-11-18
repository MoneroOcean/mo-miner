"use strict";

// Relocatable installed-SDK transport. The producer performs full LLVM admission while its
// original source/build/cache closure is mounted; consumers validate this separately pinned
// installed prefix and receipt chain without dereferencing those historical absolute paths.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"),
  {isDeepStrictEqual} = require("node:util");
const {sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {archiveName, archiveSha256} = require("./native-llvm-sdk");
const llvmBuilder = require("./build-native-llvm-sdk");

const repo = path.resolve(__dirname, ".."), format = "mom-linux-llvm-sdk-handoff-v1",
  role = "linux-llvm21-installed-sdk", receiptNames = ["result.json", "configured.json", "install.json", "build-complete.json"],
  recipeFiles = ["scripts/native-linux-llvm-sdk-handoff.js", "scripts/build-native-llvm-sdk.js", "scripts/native-llvm-sdk.js",
    "scripts/native-audit.js", "scripts/build-source-inputs.js"],
  maxFiles = 10000, maxDepth = 32, maxMemberBytes = 512 << 20, maxSdkBytes = 2 * 1024 ** 3,
  maxReceiptBytes = 8 << 20, maxManifestBytes = 16 << 20, reserveBytes = 2 * 1024 ** 3;
/** @typedef {{dev:bigint,ino:bigint,nlink:bigint,size:bigint,mtimeNs:bigint,ctimeNs:bigint,mode:bigint}} FsIdentity */

/** @param {string} value */
function hasControl(value) {return [...value].some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127);}

/** @param {unknown} name @returns {name is string} */
function safeRelative(name) {
  return typeof name === "string" && name.length > 0 && name.length <= 2048 && !name.startsWith("/") && !name.includes("\\") &&
    !hasControl(name) && name.split("/").every(part => part && part !== "." && part !== "..");
}

/** @param {string} root @param {string} file */
function contained(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep);
}

/** @param {string} left @param {string} right */
function overlaps(left, right) {return contained(left, right) || contained(right, left);}

/** @param {string} directory @param {string} label */
function canonicalDirectory(directory, label) {
  if (typeof directory !== "string" || !path.isAbsolute(directory) || path.resolve(directory) !== directory || /[\r\n\0]/.test(directory)) {
    throw new Error(label + " must be an absolute canonical path");
  }
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " must be a real directory");
  }
  return directory;
}

/** @param {string} file @param {number} maximum @param {string} label */
function stableRegular(file, maximum, label) {
  const beforePath = fs.lstatSync(file, {bigint: true}), mode = Number(beforePath.mode & 0o7777n);
  if (!beforePath.isFile() || beforePath.isSymbolicLink() || beforePath.nlink !== 1n || beforePath.size > BigInt(maximum) ||
      (mode & 0o7022) !== 0 || fs.realpathSync(file) !== file) {
    throw new Error(label + " must be a bounded canonical singly-linked regular file");
  }
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0));
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!sameStat(beforePath, before)) {throw new Error(label + " changed before hashing");}
    const hash = createHash("sha256"), buffer = Buffer.allocUnsafe(64 << 10);
    let bytes = 0n;
    while (bytes <= before.size) {
      const remaining = before.size - bytes + 1n,
        count = fs.readSync(fd, buffer, 0, Number(remaining < BigInt(buffer.length) ? remaining : BigInt(buffer.length)), null);
      if (!count) {break;}
      bytes += BigInt(count); hash.update(buffer.subarray(0, count));
    }
    const after = fs.fstatSync(fd, {bigint: true}), afterPath = fs.lstatSync(file, {bigint: true});
    if (bytes !== before.size || !sameStat(before, after) || !sameStat(before, afterPath) || fs.realpathSync(file) !== file) {
      throw new Error(label + " changed while hashing");
    }
    return {bytes: Number(before.size), sha256: hash.digest("hex"), mode,
      identity: identityOf(before)};
  } finally {fs.closeSync(fd);}
}

/** @param {string} file @param {number} maximum @param {string} label */
function stableJson(file, maximum, label) {
  const pin = stableRegular(file, maximum, label), fd = fs.openSync(file,
    fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0));
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!sameIdentity(before, pin.identity)) {throw new Error(label + " changed before parsing");}
    const buffer = Buffer.alloc(Number(before.size) + 1);
    let bytes = 0;
    while (bytes < buffer.length) {
      const count = fs.readSync(fd, buffer, bytes, buffer.length - bytes, null);
      if (!count) {break;}
      bytes += count;
    }
    const after = fs.fstatSync(fd, {bigint: true}), afterPath = fs.lstatSync(file, {bigint: true}), content = buffer.subarray(0, bytes);
    if (bytes !== pin.bytes || !sameIdentity(before, identityOf(after)) || !sameIdentity(before, identityOf(afterPath)) ||
        createHash("sha256").update(content).digest("hex") !== pin.sha256 || fs.realpathSync(file) !== file) {
      throw new Error(label + " changed while parsing");
    }
    return {value: JSON.parse(content.toString("utf8")), pin};
  } finally {fs.closeSync(fd);}
}

/** @param {string} root @param {string} file @param {string} label */
function stableSymlink(root, file, label) {
  const before = fs.lstatSync(file, {bigint: true}), target = fs.readlinkSync(file), resolved = path.resolve(path.dirname(file), target);
  if (!before.isSymbolicLink() || before.nlink !== 1n || !target || hasControl(target) || path.isAbsolute(target) ||
      !contained(root, resolved) || !fs.existsSync(resolved) || !contained(root, fs.realpathSync(file))) {
    throw new Error(label + " must resolve to an existing path inside the installed SDK");
  }
  return {target, identity: identityOf(before)};
}

/** @param {string} root @param {string} file @param {string} target @param {FsIdentity} identity @param {string} label */
function sameSymlink(root, file, target, identity, label) {
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isSymbolicLink() || !sameIdentity(stat, identity) || fs.readlinkSync(file) !== target ||
      !contained(root, path.resolve(path.dirname(file), target)) || !contained(root, fs.realpathSync(file))) {
    throw new Error(label + " changed during handoff production");
  }
}

/** @param {string} file @param {FsIdentity} identity @param {string} label */
function sameRegular(file, identity, label) {
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.isSymbolicLink() || !sameIdentity(stat, identity) || fs.realpathSync(file) !== file) {
    throw new Error(label + " changed during handoff production");
  }
}

/** @param {import("node:fs").BigIntStats} left @param {import("node:fs").BigIntStats} right */
function sameStat(left, right) {
  return left.isFile() && right.isFile() && left.dev === right.dev && left.ino === right.ino &&
    left.nlink === right.nlink && left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs && left.mode === right.mode;
}

/** @param {import("node:fs").BigIntStats} stat @returns {FsIdentity} */
function identityOf(stat) {
  return {dev: stat.dev, ino: stat.ino, nlink: stat.nlink, size: stat.size, mtimeNs: stat.mtimeNs, ctimeNs: stat.ctimeNs, mode: stat.mode};
}

/** @param {import("node:fs").BigIntStats} stat @param {FsIdentity} identity */
function sameIdentity(stat, identity) {
  return stat.dev === identity.dev && stat.ino === identity.ino && stat.nlink === identity.nlink && stat.size === identity.size &&
    stat.mtimeNs === identity.mtimeNs && stat.ctimeNs === identity.ctimeNs && stat.mode === identity.mode;
}

/** @param {string} root @returns {{files:any[],directories:string[],bytes:number,identities:Map<string, any>}} */
function inventorySdk(root) {
  const files = /** @type {any[]} */ ([]), directories = /** @type {string[]} */ ([]), identities = new Map();
  let entries = 0, bytes = 0;
  /** @param {string} directory @param {string} relative @param {number} depth */
  const visit = (directory, relative, depth) => {
    if (depth > maxDepth) {throw new Error("Linux LLVM SDK handoff tree is too deeply nested");}
    const dirStat = fs.lstatSync(directory);
    if (!dirStat.isDirectory() || dirStat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Linux LLVM SDK handoff contains an aliased directory");
    }
    if (relative) {directories.push(relative);}
    for (const name of fs.readdirSync(directory).sort()) {
      if (!name || name === "." || name === ".." || name.includes("\\") || hasControl(name) || ++entries > maxFiles * 2) {
        throw new Error("Linux LLVM SDK handoff inventory exceeds its entry bound");
      }
      const file = path.join(directory, name), member = relative ? relative + "/" + name : name;
      if (!safeRelative(member)) {throw new Error("Linux LLVM SDK handoff contains an unsafe member path");}
      const stat = fs.lstatSync(file);
      if (stat.isDirectory() && !stat.isSymbolicLink()) {visit(file, member, depth + 1); continue;}
      if (stat.isSymbolicLink()) {
        const link = stableSymlink(root, file, "LLVM SDK alias " + member);
        files.push({path: member, type: "symlink", target: link.target}); identities.set(member, link);
        bytes += Buffer.byteLength(link.target);
      } else if (stat.isFile()) {
        const pin = stableRegular(file, maxMemberBytes, "LLVM SDK file " + member);
        files.push({path: member, type: "file", bytes: pin.bytes, sha256: pin.sha256, mode: pin.mode}); identities.set(member, pin);
        bytes += pin.bytes;
      } else {throw new Error("Linux LLVM SDK handoff contains a special file");}
      if (!Number.isSafeInteger(bytes) || bytes > maxSdkBytes || files.length > maxFiles) {
        throw new Error("Linux LLVM SDK handoff exceeds its inventory or byte bound");
      }
    }
  };
  visit(root, "", 0);
  files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  directories.sort();
  return {files, directories, bytes, identities};
}

/** @param {string} root @param {number} [maxFilesAllowed] */
function bundleInventory(root, maxFilesAllowed = maxFiles + receiptNames.length + 1) {
  const files = /** @type {Array<{path:string,type:"file"|"symlink"}>} */ ([]), directories = /** @type {string[]} */ ([]);
  let entries = 0;
  /** @param {string} directory @param {string} relative @param {number} depth */
  const visit = (directory, relative, depth) => {
    if (depth > maxDepth) {throw new Error("Linux LLVM SDK bundle is too deeply nested");}
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Linux LLVM SDK bundle contains an aliased directory");
    }
    if (relative) {directories.push(relative);}
    for (const name of fs.readdirSync(directory).sort()) {
      if (!name || name === "." || name === ".." || name.includes("\\") || hasControl(name) || ++entries > maxFilesAllowed * 2 + 64) {
        throw new Error("Linux LLVM SDK bundle inventory exceeds its entry bound");
      }
      const file = path.join(directory, name), member = relative ? relative + "/" + name : name;
      if (!safeRelative(member)) {throw new Error("Linux LLVM SDK bundle contains an unsafe path");}
      const item = fs.lstatSync(file);
      if (item.isDirectory() && !item.isSymbolicLink()) {visit(file, member, depth + 1); continue;}
      if (item.isSymbolicLink()) {
        stableSymlink(root, file, "Bundle alias " + member);
        files.push({path: member, type: "symlink"});
      } else if (item.isFile()) {
        files.push({path: member, type: "file"});
      } else {
        throw new Error("Linux LLVM SDK bundle contains a special file");
      }
      if (files.length > maxFilesAllowed) {throw new Error("Linux LLVM SDK bundle contains too many files");}
    }
  };
  visit(root, "", 0);
  return {files: files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0), directories: directories.sort()};
}

/** @param {unknown} value @returns {value is any} */
function object(value) {return !!value && typeof value === "object" && !Array.isArray(value);}

/** @param {unknown} value @param {string[]} keys */
function exactKeys(value, keys) {return object(value) && Object.keys(value).sort().join("\0") === keys.slice().sort().join("\0");}

/** @param {string} output @param {any} receipts @param {Record<string, {sha256:string}>} pins
 * @param {{resultSha256:string,runId:string}} identity */
function verifyOriginalReceiptChain(output, receipts, pins, identity) {
  const result = receipts["result.json"], configured = receipts["configured.json"], install = receipts["install.json"], complete = receipts["build-complete.json"],
    plan = result?.plan, sdk = result?.sdk, snapshot = sdk?.snapshot, inputArchive = object(result?.sourceInputs) ? result.sourceInputs[plan?.archive] : undefined;
  if (!result || !configured || !install || !complete || !plan || !sdk || !snapshot || !object(snapshot.files) ||
      result.version !== 1 || result.role !== "linux-llvm21-source-sdk" || result.stage !== "audited" || result.runId !== identity.runId ||
      result.buildOnly !== true || result.compiled !== true || result.executed !== false || result.publicationQualified !== false ||
      result.archiveSha256 !== archiveSha256 || result.plan?.output !== output || result.plan?.prefix !== path.join(output, "sdk") ||
      result.sdk.path !== result.plan.prefix || path.basename(plan.archive || "") !== archiveName ||
      ![plan.output, plan.prefix, plan.source, plan.build, plan.archive].every(value => typeof value === "string" && path.isAbsolute(value) && path.normalize(value) === value) ||
      plan.source === plan.build ||
      configured.version !== 1 || configured.role !== result.role || configured.stage !== "configured" || configured.runId !== result.runId ||
      configured.archiveSha256 !== archiveSha256 ||
      configured.planSha256 !== result.planSha256 || !isDeepStrictEqual(configured.plan, plan) ||
      !isDeepStrictEqual(configured.configure, result.configure) || !isDeepStrictEqual(configured.sourceState, result.sourceState) ||
      !isDeepStrictEqual(configured.inputFiles, result.sourceInputs) || !isDeepStrictEqual(configured.toolIdentity, result.toolIdentity) ||
      inputArchive !== archiveSha256 || !Number.isSafeInteger(result.sdk.files) || result.sdk.files !== Object.keys(snapshot.files).length ||
      !Number.isSafeInteger(result.sdk.bytes) || result.sdk.bytes !== snapshot.bytes || !Number.isSafeInteger(result.archives) || result.archives < 1 ||
      result.buildComplete?.path !== path.join(output, "build-complete.json") || result.install?.path !== path.join(output, "install.json") ||
      result.buildComplete.sha256 !== pins["build-complete.json"]?.sha256 ||
      result.install.sha256 !== pins["install.json"]?.sha256 ||
      install.version !== 1 || install.role !== result.role || install.runId !== result.runId ||
      install.buildCompleteSha256 !== pins["build-complete.json"]?.sha256 ||
      !isDeepStrictEqual(install.audit, {files: result.sdk.files, bytes: result.sdk.bytes, archives: result.archives, snapshot}) ||
      complete.version !== 1 || complete.role !== result.role || complete.runId !== result.runId ||
      !Array.isArray(complete.phases) || complete.phases.length < 1) {
    throw new Error("Linux LLVM installed-SDK handoff has inconsistent original receipt links");
  }
  if (pins["result.json"]?.sha256 !== identity.resultSha256) {
    throw new Error("Linux LLVM result receipt differs from its handoff identity");
  }
  return {result, snapshot};
}

/** @param {string} admittedOutput @param {string} outputRoot
 * @param {(output:string)=>any} [reader] Internal small-fixture seam; production uses full public admission. */
function exportLinuxLlvmSdk(admittedOutput, outputRoot, reader = llvmBuilder.verifyLinuxLlvmAdmission) {
  const sourceOutput = canonicalDirectory(admittedOutput, "Admitted Linux LLVM output"),
    sourceSdk = path.join(sourceOutput, "sdk"), outputParent = canonicalDirectory(path.dirname(outputRoot), "Linux LLVM SDK handoff parent");
  if (!path.isAbsolute(outputRoot) || path.resolve(outputRoot) !== outputRoot || fs.existsSync(outputRoot) ||
      overlaps(sourceOutput, outputRoot)) {
    throw new Error("Linux LLVM installed-SDK handoff requires a fresh disjoint output");
  }
  const admitted = reader(sourceOutput), resultPin = stableJson(path.join(sourceOutput, "result.json"), maxReceiptBytes, "Original LLVM result receipt"),
    result = resultPin.value;
  if (!isDeepStrictEqual(admitted, result)) {throw new Error("Full public LLVM admission result changed before handoff");}
  const receiptPins = /** @type {Record<string, ReturnType<typeof stableRegular>>} */ ({}), receipts = /** @type {any} */ ({});
  receiptPins["result.json"] = resultPin.pin;
  receipts["result.json"] = result;
  for (const name of receiptNames.slice(1)) {
    const receipt = stableJson(path.join(sourceOutput, name), maxReceiptBytes, "Original LLVM " + name);
    receiptPins[name] = receipt.pin; receipts[name] = receipt.value;
  }
  const {snapshot} = verifyOriginalReceiptChain(sourceOutput, receipts, receiptPins,
    {resultSha256: resultPin.pin.sha256, runId: result.runId});
  if (result.sdk.path !== sourceSdk || fs.realpathSync(sourceSdk) !== sourceSdk) {
    throw new Error("Full public LLVM admission does not identify its canonical installed SDK prefix");
  }
  const inventory = inventorySdk(sourceSdk);
  verifySnapshot(snapshot, inventory.files, inventory.bytes, result.sdk);
  const receiptTotal = Object.values(receiptPins).reduce((sum, row) => sum + row.bytes, 0),
    totalBytes = inventory.bytes + receiptTotal, requiredBytes = 3 * totalBytes + reserveBytes;
  if (!Number.isSafeInteger(totalBytes) || totalBytes > maxSdkBytes + receiptNames.length * maxReceiptBytes ||
      !Number.isSafeInteger(requiredBytes) || availableBytes(outputParent) < requiredBytes) {
    throw new Error("Linux LLVM installed-SDK handoff exceeds its payload or disk-space bound");
  }
  const recipe = sourceInputs(repo, [], recipeFiles);
  fs.mkdirSync(outputRoot, {mode: 0o700}); fs.mkdirSync(path.join(outputRoot, "sdk"), {mode: 0o700});
  for (const directory of inventory.directories) {
    fs.mkdirSync(path.join(outputRoot, "sdk", ...directory.split("/")), {recursive: true, mode: 0o700});
  }
  for (const row of inventory.files) {
    const destination = path.join(outputRoot, "sdk", ...row.path.split("/"));
    fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o700});
    if (row.type === "file") {
      const sourcePin = /** @type {ReturnType<typeof stableRegular>} */ (inventory.identities.get(row.path));
      if (!sourcePin) {throw new Error("LLVM SDK file disappeared from the stable inventory");}
      fs.copyFileSync(path.join(sourceSdk, ...row.path.split("/")), destination, fs.constants.COPYFILE_EXCL);
      fs.chmodSync(destination, row.mode);
      const copied = stableRegular(destination, maxMemberBytes, "Copied LLVM SDK file " + row.path);
      if (copied.bytes !== row.bytes || copied.sha256 !== row.sha256 || copied.mode !== row.mode) {
        throw new Error("Copied LLVM SDK file differs from its admitted snapshot: " + row.path);
      }
    }
  }
  for (const row of inventory.files) {
    if (row.type !== "symlink") {continue;}
    const destination = path.join(outputRoot, "sdk", ...row.path.split("/"));
    fs.symlinkSync(row.target, destination);
  }
  for (const row of inventory.files) {
    if (row.type !== "symlink") {continue;}
    const source = path.join(sourceSdk, ...row.path.split("/")), destination = path.join(outputRoot, "sdk", ...row.path.split("/")),
      copied = stableSymlink(path.join(outputRoot, "sdk"), destination, "Copied LLVM SDK alias " + row.path),
      sourcePin = /** @type {ReturnType<typeof stableSymlink>} */ (inventory.identities.get(row.path));
    if (copied.target !== row.target) {throw new Error("Copied LLVM SDK alias changed: " + row.path);}
    if (!sourcePin) {throw new Error("LLVM SDK alias disappeared from the stable inventory");}
    sameSymlink(sourceSdk, source, row.target, sourcePin.identity, "LLVM SDK alias " + row.path);
  }
  for (const name of receiptNames) {
    const source = path.join(sourceOutput, name), destination = path.join(outputRoot, name), pin = receiptPins[name];
    if (!pin) {throw new Error("Original LLVM receipt disappeared");}
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL); fs.chmodSync(destination, pin.mode);
    const copied = stableRegular(destination, maxReceiptBytes, "Copied LLVM receipt " + name);
    if (copied.sha256 !== pin.sha256 || copied.bytes !== pin.bytes || copied.mode !== pin.mode) {
      throw new Error("Original LLVM receipt changed during handoff: " + name);
    }
    sameRegular(source, pin.identity, "Original LLVM receipt " + name);
  }
  for (const row of inventory.files) {
    const source = path.join(sourceSdk, ...row.path.split("/")), pin = inventory.identities.get(row.path);
    if (row.type === "file") {
      sameRegular(source, pin.identity, "LLVM SDK file " + row.path);
    } else {
      sameSymlink(sourceSdk, source, row.target, pin.identity, "LLVM SDK alias " + row.path);
    }
  }
  const finalSource = inventorySdk(sourceSdk);
  verifySnapshot(snapshot, finalSource.files, finalSource.bytes, result.sdk);
  if (!isDeepStrictEqual({files: finalSource.files, directories: finalSource.directories, bytes: finalSource.bytes},
    {files: inventory.files, directories: inventory.directories, bytes: inventory.bytes})) {
    throw new Error("Original LLVM SDK changed during handoff production");
  }
  const copiedSdk = inventorySdk(path.join(outputRoot, "sdk"));
  verifySnapshot(snapshot, copiedSdk.files, copiedSdk.bytes, result.sdk);
  if (!isDeepStrictEqual({files: copiedSdk.files, directories: copiedSdk.directories, bytes: copiedSdk.bytes},
    {files: inventory.files, directories: inventory.directories, bytes: inventory.bytes})) {
    throw new Error("Copied LLVM SDK changed before handoff finalization");
  }
  for (const name of receiptNames) {
    const pin = receiptPins[name], copied = stableRegular(path.join(outputRoot, name), maxReceiptBytes, "Final LLVM receipt " + name);
    if (!pin || copied.sha256 !== pin.sha256 || copied.bytes !== pin.bytes || copied.mode !== pin.mode) {
      throw new Error("Copied LLVM receipt changed before handoff finalization: " + name);
    }
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeFiles), "Linux LLVM SDK handoff recipe");
  const handoff = {version: 1, format, role, runId: result.runId,
    buildOnly: true, compiled: true, executed: false, publicationQualified: false,
    original: {output: result.plan.output, prefix: result.plan.prefix, resultSha256: receiptPins["result.json"]?.sha256},
    sourceArchive: {name: archiveName, sha256: archiveSha256, bundled: false},
    sdk: {root: "sdk", files: inventory.files.length, bytes: inventory.bytes, directories: inventory.directories},
    receipts: Object.fromEntries(receiptNames.map(name => [name, {path: name,
      bytes: receiptPins[name]?.bytes, sha256: receiptPins[name]?.sha256, mode: receiptPins[name]?.mode}])),
    files: inventory.files, recipe};
  const handoffPath = path.join(outputRoot, "handoff.json"), serialized = JSON.stringify(handoff, null, 2) + "\n";
  if (Buffer.byteLength(serialized) > maxManifestBytes) {throw new Error("Linux LLVM SDK handoff manifest exceeds its bound");}
  fs.writeFileSync(handoffPath, serialized, {flag: "wx", mode: 0o444});
  const handoffPin = stableRegular(handoffPath, maxManifestBytes, "Linux LLVM SDK handoff manifest");
  const inventoryAfter = bundleInventory(outputRoot), expectedFiles = expectedBundleFiles(handoff), expectedDirectories = expectedBundleDirectories(handoff);
  if (!isDeepStrictEqual(inventoryAfter.files, expectedFiles) || inventoryAfter.directories.join("\0") !== expectedDirectories.join("\0")) {
    throw new Error("Produced Linux LLVM SDK handoff inventory differs from its manifest");
  }
  return {handoffPath, handoffSha256: handoffPin.sha256, fileCount: inventory.files.length, bytes: inventory.bytes};
}

/** @param {unknown} snapshot @param {any[]} files @param {number} bytes @param {any} sdk */
function verifySnapshot(snapshot, files, bytes, sdk) {
  if (!object(snapshot) || !object(snapshot.files) || Object.keys(snapshot.files).length !== files.length ||
      !Number.isSafeInteger(snapshot.bytes) || snapshot.bytes !== bytes || sdk.files !== files.length || sdk.bytes !== bytes) {
    throw new Error("LLVM installed SDK inventory differs from its admitted receipt");
  }
  for (const row of files) {
    const expected = snapshot.files[row.path];
    if (!expected || row.type === "file" && (!exactKeys(expected, ["bytes", "sha256", "type"]) || expected.type !== "file" ||
        expected.bytes !== row.bytes || expected.sha256 !== row.sha256) ||
        row.type === "symlink" && (!exactKeys(expected, ["target", "type"]) || expected.type !== "symlink" || expected.target !== row.target)) {
      throw new Error("LLVM installed SDK member differs from its admitted receipt: " + row.path);
    }
  }
  if (Object.keys(snapshot.files).some(name => !files.some(row => row.path === name))) {
    throw new Error("LLVM installed SDK receipt lists a missing member");
  }
}

/** @param {any} handoff */
function expectedBundleFiles(handoff) {
  return [...handoff.files.map((/** @type {any} */ row) => ({path: "sdk/" + row.path, type: row.type})),
    ...receiptNames.map(name => ({path: name, type: "file"})), {path: "handoff.json", type: "file"}]
    .sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
}

/** @param {any} handoff */
function expectedBundleDirectories(handoff) {
  const directories = new Set(["sdk"]);
  for (const directory of handoff.sdk.directories) {directories.add("sdk/" + directory);}
  for (const row of handoff.files) {
    const parts = row.path.split("/");
    for (let index = 1; index < parts.length; index++) {directories.add("sdk/" + parts.slice(0, index).join("/"));}
  }
  return [...directories].sort();
}

/** @param {string} bundleRoot @param {{handoffSha256:string}} expectation */
function linuxLlvmSdkHandoffInputs(bundleRoot, expectation) {
  const root = canonicalDirectory(bundleRoot, "Linux LLVM SDK handoff root");
  if (!expectation || typeof expectation !== "object" || !/^[a-f0-9]{64}$/.test(expectation.handoffSha256 || "")) {
    throw new Error("Linux LLVM SDK handoff requires its external manifest pin");
  }
  const manifestPath = path.join(root, "handoff.json"), manifestPin = stableJson(manifestPath, maxManifestBytes, "Linux LLVM SDK handoff manifest"),
    handoff = manifestPin.value;
  if (!exactKeys(handoff, ["version", "format", "role", "runId", "buildOnly", "compiled", "executed", "publicationQualified",
    "original", "sourceArchive", "sdk", "receipts", "files", "recipe"]) || handoff.version !== 1 || handoff.format !== format ||
      handoff.role !== role || handoff.buildOnly !== true || handoff.compiled !== true ||
      handoff.executed !== false || handoff.publicationQualified !== false || manifestPin.pin.sha256 !== expectation.handoffSha256 ||
      typeof handoff.runId !== "string" || !/^[0-9a-f-]{36}$/i.test(handoff.runId) ||
      !exactKeys(handoff.original, ["output", "prefix", "resultSha256"]) ||
      ![handoff.original.output, handoff.original.prefix].every(value => typeof value === "string" && path.isAbsolute(value) && path.normalize(value) === value) ||
      handoff.original.prefix !== path.join(handoff.original.output, "sdk") || !/^[a-f0-9]{64}$/.test(handoff.original.resultSha256 || "") ||
      !exactKeys(handoff.sourceArchive, ["name", "sha256", "bundled"]) || handoff.sourceArchive.name !== archiveName ||
      handoff.sourceArchive.sha256 !== archiveSha256 || handoff.sourceArchive.bundled !== false ||
      !exactKeys(handoff.sdk, ["root", "files", "bytes", "directories"]) || handoff.sdk.root !== "sdk" || !Number.isSafeInteger(handoff.sdk.files) ||
      handoff.sdk.files < 1 || handoff.sdk.files > maxFiles || !Number.isSafeInteger(handoff.sdk.bytes) || handoff.sdk.bytes < 1 || handoff.sdk.bytes > maxSdkBytes ||
      !exactKeys(handoff.receipts, receiptNames) || !Array.isArray(handoff.files) || handoff.files.length !== handoff.sdk.files ||
      !object(handoff.recipe)) {
    throw new Error("Linux LLVM SDK handoff manifest is invalid or belongs to another run");
  }
  verifySourceInputs(handoff.recipe, sourceInputs(repo, [], recipeFiles), "Linux LLVM SDK handoff recipe");
  const {files: handoffFiles, bytes: handoffBytes} = validateFileRows(handoff.files);
  const sdkDirectories = validateDirectoryRows(handoff.sdk.directories, handoffFiles);
  if (handoffBytes !== handoff.sdk.bytes) {throw new Error("Linux LLVM SDK handoff byte count differs from its manifest");}
  const receiptPins = /** @type {Record<string, ReturnType<typeof stableRegular>>} */ ({}), receipts = /** @type {any} */ ({});
  for (const name of receiptNames) {
    const row = handoff.receipts[name];
    if (!exactKeys(row, ["path", "bytes", "sha256", "mode"]) || row.path !== name ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > maxReceiptBytes || !/^[a-f0-9]{64}$/.test(row.sha256 || "") ||
        !Number.isSafeInteger(row.mode) || (row.mode & 0o7022) !== 0) {
      throw new Error("Linux LLVM SDK handoff receipt pin is malformed: " + name);
    }
    const file = path.join(root, row.path), pin = stableRegular(file, maxReceiptBytes, "LLVM handoff receipt " + name);
    if (pin.bytes !== row.bytes || pin.sha256 !== row.sha256 || pin.mode !== row.mode) {
      throw new Error("Original LLVM receipt differs from its external handoff pin: " + name);
    }
    const parsed = stableJson(file, maxReceiptBytes, "LLVM handoff receipt " + name);
    if (!isDeepStrictEqual(parsed.pin.identity, pin.identity)) {throw new Error("LLVM handoff receipt changed while parsing: " + name);}
    receiptPins[name] = pin; receipts[name] = parsed.value;
  }
  const {snapshot} = verifyOriginalReceiptChain(handoff.original.output, receipts, receiptPins,
    {resultSha256: receiptPins["result.json"]?.sha256 || "", runId: handoff.runId});
  if (receiptPins["result.json"]?.sha256 !== handoff.original.resultSha256) {
    throw new Error("Original LLVM result receipt differs from the handoff identity");
  }
  const sdkRoot = path.join(root, handoff.sdk.root), inventory = inventorySdk(sdkRoot);
  verifySnapshot(snapshot, inventory.files.map(({identity: _identity, ...row}) => row), inventory.bytes, handoff.sdk);
  if (!isDeepStrictEqual(inventory.files.map(({identity: _identity, ...row}) => row), handoffFiles) ||
      inventory.directories.join("\0") !== sdkDirectories.join("\0")) {
    throw new Error("LLVM SDK package files differ from the externally pinned manifest");
  }
  const bundle = bundleInventory(root), expectedFiles = expectedBundleFiles(handoff), expectedDirectories = expectedBundleDirectories(handoff);
  if (!isDeepStrictEqual(bundle.files, expectedFiles) || bundle.directories.join("\0") !== expectedDirectories.join("\0")) {
    throw new Error("Linux LLVM SDK handoff bundle has missing or extra members");
  }
  const finalSdk = inventorySdk(sdkRoot);
  if (!isDeepStrictEqual(finalSdk.files, inventory.files) || !isDeepStrictEqual(finalSdk.identities, inventory.identities) ||
      finalSdk.directories.join("\0") !== inventory.directories.join("\0") ||
      receiptNames.some(name => {
        const pin = stableRegular(path.join(root, name), maxReceiptBytes, "Final LLVM handoff receipt " + name), expected = receiptPins[name];
        return !expected || pin.sha256 !== expected.sha256 || pin.bytes !== expected.bytes || pin.mode !== expected.mode ||
          !sameIdentity(fs.lstatSync(path.join(root, name), {bigint: true}), expected.identity);
      }) ||
      stableJson(manifestPath, maxManifestBytes, "Linux LLVM SDK handoff manifest").pin.sha256 !== expectation.handoffSha256 ||
      !isDeepStrictEqual(bundleInventory(root), bundle)) {
    throw new Error("Linux LLVM SDK handoff changed during admission");
  }
  const files = Object.fromEntries(handoffFiles.map(row => [row.path, row]));
  return {sdkRoot, files, runId: handoff.runId,
    resultSha256: handoff.original.resultSha256, handoffSha256: expectation.handoffSha256,
    sourceArchiveSha256: archiveSha256};
}

/** @param {unknown} rows @returns {{files:any[],bytes:number}} */
function validateFileRows(rows) {
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > maxFiles) {throw new Error("Linux LLVM SDK handoff file list is outside its bound");}
  const files = /** @type {any[]} */ ([]), seen = new Set();
  let bytes = 0;
  for (const row of rows) {
    if (!object(row) || !safeRelative(row.path) || seen.has(row.path)) {throw new Error("Linux LLVM SDK handoff contains a duplicate or unsafe file path");}
    seen.add(row.path);
    if (row.type === "file" && exactKeys(row, ["path", "type", "bytes", "sha256", "mode"]) &&
        Number.isSafeInteger(row.bytes) && row.bytes >= 0 && row.bytes <= maxMemberBytes && /^[a-f0-9]{64}$/.test(row.sha256 || "") &&
        Number.isSafeInteger(row.mode) && row.mode > 0 && row.mode <= 0o777 && (row.mode & 0o7022) === 0) {
      bytes += row.bytes;
    } else if (row.type === "symlink" && exactKeys(row, ["path", "type", "target"]) && typeof row.target === "string" &&
        row.target.length > 0 && row.target.length <= 2048 && !path.isAbsolute(row.target) && !hasControl(row.target)) {
      bytes += Buffer.byteLength(row.target);
    } else {throw new Error("Linux LLVM SDK handoff file row is malformed: " + row.path);}
    if (!Number.isSafeInteger(bytes) || bytes > maxSdkBytes) {throw new Error("Linux LLVM SDK handoff exceeds its byte bound");}
    files.push(row);
  }
  if (!isDeepStrictEqual(files.map(row => row.path), files.map(row => row.path).slice().sort())) {
    throw new Error("Linux LLVM SDK handoff file rows are not sorted");
  }
  return {files, bytes};
}

/** @param {unknown} rows @param {any[]} files @returns {string[]} */
function validateDirectoryRows(rows, files) {
  if (!Array.isArray(rows) || rows.length > maxFiles || rows.some(row => !safeRelative(row))) {
    throw new Error("Linux LLVM SDK handoff directory list is malformed");
  }
  const directories = /** @type {string[]} */ (rows), seen = new Set(directories), filePaths = new Set(files.map(row => row.path));
  if (seen.size !== directories.length || !isDeepStrictEqual(directories, directories.slice().sort()) ||
      directories.some(directory => filePaths.has(directory))) {
    throw new Error("Linux LLVM SDK handoff directory list is duplicated, unsorted, or conflicts with a file");
  }
  for (const directory of directories) {
    const parts = directory.split("/");
    for (let index = 1; index < parts.length; index++) {
      if (!seen.has(parts.slice(0, index).join("/"))) {throw new Error("Linux LLVM SDK handoff omits a parent directory");}
    }
  }
  return directories;
}

/** @param {string} directory */
function availableBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine Linux LLVM SDK handoff disk capacity");}
  const bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

module.exports = {exportLinuxLlvmSdk, linuxLlvmSdkHandoffInputs};
