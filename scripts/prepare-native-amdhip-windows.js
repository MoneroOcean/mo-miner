"use strict";

// Generate the AMD Windows import library from ROCm's pinned upstream export definition.
const fs = require("node:fs"), path = require("node:path"), childProcess = require("node:child_process");
const {createHash, randomUUID} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const sourceIO = require("./build-source-inputs");
const {ciSuffix, validCiSuffix} = require("./native-ci-job");
const {readBuildJson, sourceInputs} = sourceIO;
const localSdk = require("./build-local-native-sdk"), localRocm = require("./build-local-native-rocm-sdk"),
  {verifyLinuxLlvmAdmission} = require("./build-native-llvm-sdk"), nativeCi = require("./native-ci-linux-sources");
const repo = path.resolve(__dirname, ".."), catalogFile = path.join(repo, "runtime/windows-amdhip.json");
const catalog = /** @type {AmdhipCatalog} */ (require("../runtime/windows-amdhip.json"));
/** @param {object} value */
function freezeTree(value) {for (const child of Object.values(value)) {if (child && typeof child === "object") {freezeTree(child);}} Object.freeze(value);}
freezeTree(catalog);
const recipeFiles = ["scripts/prepare-native-amdhip-windows.js", "runtime/windows-amdhip.json", "scripts/build-source-inputs.js",
  "scripts/native-ci-linux-sources.js", "scripts/fetch-native-sdk-sources.js", "scripts/native-linux-llvm-tools.js",
  "scripts/native-llvm-sdk.js"];
const deadlineMs = 15 * 60 * 1000, maxCommandOutput = 8 << 20, maxDefinitionBytes = 64 << 10, maxLibraryBytes = 4 << 20;
const localBuildIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** @typedef {{version:number,target:string,source:{origin:string,revision:string,definition:{path:string,blob:string,bytes:number,sha256:string,derivedBytes:number,derivedSha256:string},license:{path:string,blob:string,bytes:number,sha256:string}},
 * library:{fileName:string,dllName:string,bytes:number,sha256:string},toolchain:{llvmRole:string,llvmLibAlias:string,llvmLibTarget:string}}} AmdhipCatalog */
/** @typedef {{path:string,blob:string,bytes:number,sha256:string,actualSha256:string}} AmdhipBlobProof */
/** @typedef {{path:string,blob:string,bytes:number,sha256:string,actualSha256:string,note:string}} AmdhipLicenseProof */
/** @typedef {{origin:string,revision:string,sourceRoot:string,fetchReceiptSha256:string,rootsReceiptSha256:string,
 * git:{path:string,bytes:number,sha256:string},definition:AmdhipBlobProof,license:AmdhipLicenseProof}} AmdhipSourceProof */
/** @typedef {{relative:string,aliasTarget:string,bytes:number,sha256:string}} AmdhipToolAlias */
/** @typedef {{role:string,llvmLib:AmdhipToolAlias,command:string[],handoffRoot?:string,handoffSha256?:string,sdkOutput?:string,sdkResultSha256?:string}} AmdhipToolchainProof */
/** @typedef {{path:string,sourceBlob:string,sourceSha256:string,bytes:number,derivedBytes:number,derivedSha256:string,
 * transformation:string}} AmdhipDefinitionProof */
/** @typedef {{fileName:string,dllName:string,bytes:number,sha256:string}} AmdhipLibraryProof */
/** @typedef {{kind:string,bytes:number,sha256:string,source:{kind:string,path:string,blob:string,sha256?:string}}} AmdhipFileProof */
/** @typedef {{version:2,target:"windows-amdhip64",status:"PASS",networked:false,buildOnly:true,compiled:false,executed:false,
 * vendorPrebuilt:false,sourceGenerated:true,publicationQualified:false,redistributionQualified:false,producerSha256:string,
 * localBuildId?:string,recipe:{sha256:string,files:Record<string,string>},source:AmdhipSourceProof,
 * toolchain:AmdhipToolchainProof,definition:AmdhipDefinitionProof,library:AmdhipLibraryProof,
 * output:string,files:Record<string,AmdhipFileProof>,note:string}} AmdhipReceipt */
/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
  path.resolve(value) === value && !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);
}
/** @param {unknown} value */
function provenancePath(value) {return typeof value === "string" && value.length > 0 && value.length <= 4096 && !/[\r\n\0]/.test(value);}
/** @param {unknown} value */
function safeGitPath(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 512 && !value.includes("\\") &&
    value.split("/").every(part => part && part !== "." && part !== ".." && /^[A-Za-z0-9._+-]+$/.test(part));
}
/** @param {unknown} value */
function safeLeaf(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 255 && /^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(value) &&
    !/[. ]$/.test(value) && !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(value);
}
/** @param {unknown} value */
function safeToolPath(value) {return typeof value === "string" && value.startsWith("bin/") && safeLeaf(value.slice(4));}
/** @param {string} file @param {string} label @param {number} maximum @param {bigint} [maxLinks] */
function regularSnapshot(file, label, maximum, maxLinks = 1n) {
  if (!absolute(file)) {throw new Error("Windows AMD " + label + " path must be canonical and absolute");}
  const stat = fs.lstatSync(file, {bigint: true});
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink < 1n || stat.nlink > maxLinks || stat.size < 1n ||
      stat.size > BigInt(maximum) || fs.realpathSync(file) !== file) {throw new Error("Windows AMD " + label + " must be a bounded canonical regular file: " + file);}
  return stat;
}
/** @param {import("node:fs").BigIntStats} a @param {import("node:fs").BigIntStats} b */
function sameSnapshot(a, b) {return a.dev === b.dev && a.ino === b.ino && a.nlink === b.nlink && a.size === b.size && a.mode === b.mode && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;}
/** @param {string} file @param {string} label @param {number} bytes @param {string} sha256 @param {number} maximum */
function verifyFile(file, label, bytes, sha256, maximum) {
  const before = regularSnapshot(file, label, maximum), actualSha256 = sourceIO.fileHash(file), after = regularSnapshot(file, label, maximum);
  if (!sameSnapshot(before, after) || Number(after.size) !== bytes || actualSha256 !== sha256) {throw new Error("Windows AMD " + label + " differs from its pinned bytes: " + file);}
  return {before, bytes, sha256: actualSha256};
}
/** Re-read the on-disk catalog so the require cache is never the sole trust source. @returns {AmdhipCatalog} */
function windowsAmdhipCatalog() {
  const before = regularSnapshot(catalogFile, "catalog", 32 << 10), disk = readBuildJson(catalogFile), after = regularSnapshot(catalogFile, "catalog", 32 << 10);
  const sourcePins = require("./fetch-native-sdk-sources"), clr = sourcePins.gitSourceComponents("rocm")["clr"];
  if (!clr) {throw new Error("Maintained ROCm source pins omit CLR");}
  const roles = {source: ["origin", "revision", "definition", "license"], definition: ["path", "blob", "bytes", "sha256", "derivedBytes", "derivedSha256"],
    license: ["path", "blob", "bytes", "sha256"], library: ["fileName", "dllName", "bytes", "sha256"],
    toolchain: ["llvmRole", "llvmLibAlias", "llvmLibTarget"]};
  /** @param {unknown} value */
  const digest = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value),
    valid = sourceIO.hasExactBuildFields(catalog, ["version", "target", "source", "library", "toolchain"]) && catalog.version === 2 && catalog.target === "windows-amdhip64" &&
      sourceIO.hasExactBuildFields(catalog.source, roles.source) && catalog.source.origin === sourcePins.origins.clr && catalog.source.revision === clr.revision &&
      sourceIO.hasExactBuildFields(catalog.source.definition, roles.definition) && safeGitPath(catalog.source.definition.path) &&
      /^[a-f0-9]{40}$/.test(catalog.source.definition.blob || "") && Number.isSafeInteger(catalog.source.definition.bytes) &&
      catalog.source.definition.bytes > 0 && catalog.source.definition.bytes <= 64 << 10 &&
      digest(catalog.source.definition.sha256) && Number.isSafeInteger(catalog.source.definition.derivedBytes) &&
      catalog.source.definition.derivedBytes === catalog.source.definition.bytes + Buffer.byteLength("LIBRARY " + catalog.library.dllName + "\n") &&
      digest(catalog.source.definition.derivedSha256) &&
      sourceIO.hasExactBuildFields(catalog.source.license, roles.license) && safeGitPath(catalog.source.license.path) && /^[a-f0-9]{40}$/.test(catalog.source.license.blob || "") &&
      Number.isSafeInteger(catalog.source.license.bytes) && catalog.source.license.bytes > 0 && catalog.source.license.bytes <= 1 << 20 && digest(catalog.source.license.sha256) &&
      sourceIO.hasExactBuildFields(catalog.library, roles.library) && safeLeaf(catalog.library.fileName) && catalog.library.fileName.toLowerCase().endsWith(".lib") &&
      safeLeaf(catalog.library.dllName) && catalog.library.dllName.toLowerCase().endsWith(".dll") &&
      path.posix.basename(catalog.source.license.path) !== catalog.library.fileName && path.posix.basename(catalog.source.license.path) !== "sdk.json" &&
      Number.isSafeInteger(catalog.library.bytes) && catalog.library.bytes > 0 && catalog.library.bytes <= maxLibraryBytes && digest(catalog.library.sha256) &&
      sourceIO.hasExactBuildFields(catalog.toolchain, roles.toolchain) && /^[a-z0-9][a-z0-9-]{1,80}$/.test(catalog.toolchain.llvmRole || "") &&
      safeToolPath(catalog.toolchain.llvmLibAlias) && safeLeaf(catalog.toolchain.llvmLibTarget) &&
      catalog.toolchain.llvmLibAlias !== "bin/" + catalog.toolchain.llvmLibTarget &&
      path.posix.basename(catalog.toolchain.llvmLibAlias) !== catalog.toolchain.llvmLibTarget;
  if (!sameSnapshot(before, after) || !isDeepStrictEqual(disk, catalog) || !valid) {
    throw new Error("Pinned Windows AMD import-library catalog is invalid");
  }
  return catalog;
}
/** @returns {ReturnType<typeof sourceInputs>} */
function amdhipRecipe() {return sourceInputs(repo, [], recipeFiles);}

/** @param {NodeJS.ProcessEnv} env */
function runPaths(env) {
  const suffix = ciSuffix(env), runnerTemp = env["RUNNER_TEMP"] || "", rocmRoot = "/tmp/mom-native-rocm-prerequisites-" + suffix;
  if (!absolute(runnerTemp) || fs.realpathSync(runnerTemp) !== runnerTemp ||
      env["MOM_AOT_BUILD_ONLY"] !== "1" || !["2", "4"].includes(env["MOM_BUILD_JOBS"] || "") || env["MOM_NATIVE_ROCM_SOURCE_DIR"] !== rocmRoot) {
    throw new Error("Windows AMD generation needs the exact run-scoped ROCm source root and build-only CI environment");
  }
  const output = path.join(runnerTemp, "native-windows-amdhip-" + suffix);
  return {suffix, runnerTemp, source: rocmRoot, output};
}
/** @param {string} output @param {string} runnerTemp @param {string} expected */
function newDestination(output, runnerTemp, expected) {
  if (!absolute(output) || output !== expected || path.dirname(output) !== runnerTemp || output.startsWith(repo + path.sep)) {
    throw new Error("Windows AMD output must be the exact fresh run-scoped artifact directory");
  }
  const stat = fs.lstatSync(output, {throwIfNoEntry: false});
  if (stat) {throw new Error("Windows AMD output must not already exist");}
  return output;
}

/** Add only the DLL directive required by llvm-lib; leave every source-definition byte intact.
 * @param {Buffer} definition @param {string} dllName */
function deriveDefinition(definition, dllName) {
  if (!Buffer.isBuffer(definition) || definition.length > maxDefinitionBytes || definition.includes(0) || !safeLeaf(dllName) || !dllName.toLowerCase().endsWith(".dll")) {
    throw new Error("Pinned AMD definition or DLL name is invalid or excessive");
  }
  return Buffer.concat([Buffer.from("LIBRARY " + dllName + "\n", "ascii"), definition]);
}

/** Admit the externally pinned tool-only handoff; never re-admit the full LLVM SDK here.
 * @param {string} bundleRoot @param {string} handoffSha256 */
function verifyLinuxToolHandoff(bundleRoot, handoffSha256) {
  if (!/^[a-f0-9]{64}$/.test(handoffSha256 || "")) {throw new Error("Windows AMD Linux LLVM handoff SHA-256 is invalid");}
  const input = require("./native-linux-llvm-tools").linuxLlvmToolHandoffInputs(bundleRoot, {handoffSha256}),
    root = path.dirname(input.toolRoot), files = input.files, aliasRelative = catalog.toolchain.llvmLibAlias,
    targetRelative = "bin/" + catalog.toolchain.llvmLibTarget,
    aliasRow = files[aliasRelative], targetRow = files[targetRelative];
  if (aliasRow?.type !== "symlink" || aliasRow.target !== catalog.toolchain.llvmLibTarget || targetRow?.type !== "file") {
    throw new Error("Windows AMD Linux LLVM tool handoff is stale or does not contain the admitted tool aliases");
  }
  return {input, root, llvmLib: path.join(input.toolRoot, path.basename(aliasRelative)),
    llvmLibIdentity: {relative: aliasRelative, aliasTarget: catalog.toolchain.llvmLibTarget, bytes: targetRow.bytes, sha256: targetRow.sha256}};
}

/** Run one bounded LLVM command; tool output is never treated as an admission record. @param {string} file @param {string[]} args @param {string} cwd @param {number} deadline */
function runTool(file, args, cwd, deadline) {
  const remaining = Math.min(5 * 60 * 1000, deadline - Date.now());
  if (remaining <= 0) {throw new Error("Windows AMD import-library generation deadline exceeded");}
  const result = childProcess.spawnSync(file, args, {cwd, env: {PATH: "/usr/bin:/bin", LC_ALL: "C", LANG: "C", TZ: "UTC", TMPDIR: cwd},
    encoding: "utf8", timeout: remaining, maxBuffer: maxCommandOutput, windowsHide: true});
  const stdout = result.stdout || "", stderr = result.stderr || "";
  if (result.error || result.signal || result.status !== 0 || Buffer.byteLength(stdout) + Buffer.byteLength(stderr) > maxCommandOutput || stdout.includes("\0") || stderr.includes("\0")) {
    throw new Error("Windows AMD " + path.basename(file) + " failed" + (result.error ? ": " + result.error.message :
      result.signal ? ": signal " + result.signal : ": exit " + result.status) + "; " + stderr.slice(0, 2048));
  }
  return stdout;
}

/** Read one CLR blob by its locked revision, tree object ID, size and digest.
 * @param {string} sourceRoot @param {string} git @param {string} revision @param {string} relative @param {string} blob @param {number} size @param {string} hash */
function readPinnedClrBlob(sourceRoot, git, revision, relative, blob, size, hash) {
  const gitEnv = {PATH: "/usr/bin:/bin", LC_ALL: "C", LANG: "C", TZ: "UTC", GIT_ALLOW_PROTOCOL: "file", GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_CONFIG_NOSYSTEM: "1", GIT_NO_REPLACE_OBJECTS: "1", GIT_OPTIONAL_LOCKS: "0", GIT_TERMINAL_PROMPT: "0"},
    gitDirectory = path.join(sourceRoot, "roots/clr");
  const result = childProcess.spawnSync(git, ["--git-dir=" + gitDirectory, "show", revision + ":" + relative],
      {env: gitEnv, encoding: "buffer", timeout: 30_000, maxBuffer: size + 1}), bytes = result.stdout;
  if (result.error || result.signal || result.status !== 0 || !Buffer.isBuffer(bytes) || bytes.length !== size ||
      bytes.includes(0) && relative.endsWith(".def") || createHash("sha256").update(bytes).digest("hex") !== hash) {
    throw new Error("Pinned CLR " + relative + " Git blob differs from its upstream object");
  }
  const check = childProcess.spawnSync(git, ["--git-dir=" + gitDirectory, "rev-parse", "--verify", revision + ":" + relative],
    {env: gitEnv, encoding: "utf8", timeout: 30_000, maxBuffer: 4096});
  if (check.error || check.signal || check.status !== 0 || check.stdout.trim() !== blob) {throw new Error("Pinned CLR Git blob object identity changed: " + relative);}
  return bytes;
}

/** Produce the shared sealed payload from the same pinned blobs and LLVM tool for CI and local callers.
 * @param {{selected:AmdhipCatalog,sourceRoot:string,destination:string,stagingParent:string,stagingPrefix:string,
 * identity:{localBuildId?:string},toolchain:Omit<AmdhipToolchainProof,"command">,toolPath:string,fetchReceiptSha256:string,rootsReceiptSha256:string,
 * recipe:ReturnType<typeof amdhipRecipe>,recipeSha256:string,producerSha256:string,verifyInputs:()=>unknown}} input */
async function generateWindowsAmdhipPayload(input) {
  const {selected, sourceRoot, destination, stagingParent, stagingPrefix} = input, deadline = Date.now() + deadlineMs,
    git = "/usr/bin/git", gitStat = regularSnapshot(git, "Git", 64 << 20);
  if ((gitStat.mode & 0o111n) === 0n) {throw new Error("Pinned source Git verifier is not executable");}
  const sourceGit = {path: git, bytes: Number(gitStat.size), sha256: sourceIO.fileHash(git)},
    definition = readPinnedClrBlob(sourceRoot, git, selected.source.revision, selected.source.definition.path, selected.source.definition.blob,
      selected.source.definition.bytes, selected.source.definition.sha256),
    license = readPinnedClrBlob(sourceRoot, git, selected.source.revision, selected.source.license.path, selected.source.license.blob,
      selected.source.license.bytes, selected.source.license.sha256);
  if (Date.now() >= deadline) {throw new Error("Windows AMD import-library generation deadline exceeded");}
  const staging = path.join(stagingParent, stagingPrefix + "-" + randomUUID());
  fs.mkdirSync(staging, {mode: 0o700});
  let published = false;
  try {
    const licenseName = path.posix.basename(selected.source.license.path), definitionPath = path.join(staging, "amdhip.def"),
      libraryPath = path.join(staging, selected.library.fileName), licensePath = path.join(staging, licenseName), derivedDefinition = deriveDefinition(definition, selected.library.dllName);
    if (derivedDefinition.length !== selected.source.definition.derivedBytes ||
        createHash("sha256").update(derivedDefinition).digest("hex") !== selected.source.definition.derivedSha256) {
      throw new Error("Derived AMD definition differs from the pinned one-line transformation");
    }
    fs.writeFileSync(definitionPath, derivedDefinition, {flag: "wx", mode: 0o600});
    fs.writeFileSync(licensePath, license, {flag: "wx", mode: 0o600});
    runTool(input.toolPath, ["/def:" + definitionPath, "/machine:x64", "/out:" + libraryPath], staging, deadline);
    fs.unlinkSync(definitionPath);
    const archiveStat = regularSnapshot(libraryPath, "generated import library", maxLibraryBytes), librarySha256 = sourceIO.fileHash(libraryPath);
    if (Number(archiveStat.size) !== selected.library.bytes || librarySha256 !== selected.library.sha256) {
      throw new Error("Generated AMD import library differs from the independently verified output pin");
    }
    fs.chmodSync(libraryPath, 0o444); fs.chmodSync(licensePath, 0o444);
    const definitionSha256 = createHash("sha256").update(definition).digest("hex"), derivedDefinitionSha256 = createHash("sha256").update(derivedDefinition).digest("hex"),
      licenseSha256 = createHash("sha256").update(license).digest("hex"), finalFiles = {
        [selected.library.fileName]: {kind: "file", bytes: selected.library.bytes, sha256: librarySha256,
          source: {kind: "generated-from-pinned-definition", path: selected.source.definition.path, blob: selected.source.definition.blob, sha256: definitionSha256}},
        [licenseName]: {kind: "file", bytes: selected.source.license.bytes, sha256: licenseSha256,
          source: {kind: "original-source-license-blob", path: selected.source.license.path, blob: selected.source.license.blob}}
      }, sourceProof = {origin: selected.source.origin, revision: selected.source.revision, sourceRoot,
        fetchReceiptSha256: input.fetchReceiptSha256, rootsReceiptSha256: input.rootsReceiptSha256, git: sourceGit,
        definition: {path: selected.source.definition.path, blob: selected.source.definition.blob, bytes: definition.length,
          sha256: selected.source.definition.sha256, actualSha256: definitionSha256},
        license: {...selected.source.license, actualSha256: licenseSha256,
          note: "Original upstream license text is preserved as evidence; no legal conclusion is asserted."}},
      toolchain = {...input.toolchain, command: ["/def:<staging>/amdhip.def", "/machine:x64", "/out:<staging>/" + selected.library.fileName]},
      receipt = {version: 2, target: "windows-amdhip64", status: "PASS", networked: false,
        buildOnly: true, compiled: false, executed: false, vendorPrebuilt: false, sourceGenerated: true,
        publicationQualified: false, redistributionQualified: false, producerSha256: input.producerSha256, ...input.identity,
        recipe: {sha256: input.recipeSha256, files: input.recipe.graph}, source: sourceProof, toolchain,
        definition: {path: selected.source.definition.path, sourceBlob: selected.source.definition.blob, sourceSha256: definitionSha256,
          bytes: definition.length, derivedBytes: derivedDefinition.length, derivedSha256: derivedDefinitionSha256,
          transformation: "Only a LIBRARY directive is prepended to the unchanged upstream .def bytes."},
        library: {fileName: selected.library.fileName, dllName: selected.library.dllName, bytes: Number(archiveStat.size), sha256: librarySha256},
        output: destination, files: finalFiles,
        note: "Source-generated Windows AMD import library only; build-only and not publication or redistribution qualified."};
    fs.writeFileSync(path.join(staging, "sdk.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx", mode: 0o444});
    fs.chmodSync(staging, 0o555);
    input.verifyInputs();
    fs.renameSync(staging, destination); published = true;
    admitWindowsAmdhip(destination);
    return receipt;
  } catch (error) {
    const owned = published ? destination : staging;
    try {
      const stat = fs.lstatSync(owned, {throwIfNoEntry: false});
      if (stat?.isDirectory() && !stat.isSymbolicLink()) {fs.chmodSync(owned, 0o700); fs.rmSync(owned, {recursive: true});}
    } catch { /* Keep the original generation failure if owned output cleanup is unavailable. */ }
    throw error;
  }
}

/** @param {string} sourceRoot @param {string} toolBundleRoot @param {string} toolHandoffSha256 @param {string} output @param {NodeJS.ProcessEnv} [environment] */
async function prepareWindowsAmdhip(sourceRoot, toolBundleRoot, toolHandoffSha256, output, environment = process.env) {
  const selected = windowsAmdhipCatalog();
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows AMD import-library generation requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const paths = runPaths(environment), destination = newDestination(output, paths.runnerTemp, paths.output);
  if (sourceRoot !== paths.source || !absolute(sourceRoot) || !absolute(toolBundleRoot) || !/^[a-f0-9]{64}$/.test(toolHandoffSha256 || "")) {
    throw new Error("Windows AMD inputs must be the original ROCm source and externally pinned Linux LLVM tool handoff");
  }
  const recipe = amdhipRecipe(), recipeSha256 = createHash("sha256").update(JSON.stringify(recipe)).digest("hex"),
    producerSha256 = sourceIO.fileHash(__filename), ci = require("./native-ci-linux-sources"),
    ciRunPaths = ci.ciPaths(), sourceState = ci.rocmSourceStatePaths(ciRunPaths, environment),
    git = "/usr/bin/git";
  if (ciRunPaths.suffix !== paths.suffix || ciRunPaths.runnerTemp !== paths.runnerTemp || sourceState.source !== sourceRoot) {
    throw new Error("ROCm source admission path differs from the run-scoped root");
  }
  ci.verifyRocmSources(ciRunPaths, sourceState, git);
  const rootsBefore = sourceIO.fileHash(sourceState.rootsFile), sourceReceiptSha256 = sourceIO.fileHash(sourceState.fetchReceipt),
    tools = verifyLinuxToolHandoff(toolBundleRoot, toolHandoffSha256),
    toolchain = {role: selected.toolchain.llvmRole, handoffRoot: tools.root, handoffSha256: toolHandoffSha256, llvmLib: tools.llvmLibIdentity},
    verifyInputs = () => {
      ci.verifyRocmSources(ciRunPaths, sourceState, git);
      const toolsAfter = verifyLinuxToolHandoff(toolBundleRoot, toolHandoffSha256);
      if (sourceIO.fileHash(sourceState.rootsFile) !== rootsBefore || sourceIO.fileHash(sourceState.fetchReceipt) !== sourceReceiptSha256 ||
          !isDeepStrictEqual(toolsAfter.llvmLibIdentity, tools.llvmLibIdentity) || sourceIO.fileHash(__filename) !== producerSha256 ||
          !isDeepStrictEqual(amdhipRecipe(), recipe) || !isDeepStrictEqual(windowsAmdhipCatalog(), selected)) {
        throw new Error("Windows AMD source, tool, catalog or producer recipe changed during generation");
      }
    };
  return generateWindowsAmdhipPayload({selected, sourceRoot, destination, stagingParent: paths.runnerTemp,
    stagingPrefix: ".native-windows-amdhip-" + paths.suffix, identity: {}, toolchain,
    toolPath: tools.llvmLib, fetchReceiptSha256: sourceReceiptSha256, rootsReceiptSha256: rootsBefore,
    recipe, recipeSha256, producerSha256, verifyInputs});
}

/** Admit the caller's Linux LLVM SDK and exact llvm-lib alias without relying on CI transport.
 * @param {{llvmOutput:string,llvmLib:string}} input @param {typeof verifyLinuxLlvmAdmission} verify */
function verifyLocalAmdhipTool(input, verify = verifyLinuxLlvmAdmission) {
  const admitted = verify(input.llvmOutput), prefix = admitted.plan?.prefix, resultFile = path.join(input.llvmOutput, "result.json"),
    alias = path.join(prefix || "", catalog.toolchain.llvmLibAlias), target = path.join(prefix || "", "bin", catalog.toolchain.llvmLibTarget),
    snapshot = admitted.sdk?.snapshot?.files, aliasRow = snapshot?.[catalog.toolchain.llvmLibAlias], targetRow = snapshot?.["bin/" + catalog.toolchain.llvmLibTarget];
  if (!absolute(prefix) || admitted.plan.output !== input.llvmOutput || admitted.sdk?.path !== prefix || input.llvmLib !== alias ||
      !snapshot || aliasRow?.type !== "symlink" || aliasRow.target !== catalog.toolchain.llvmLibTarget || targetRow?.type !== "file") {
    throw new Error("Local AMD import-library tool differs from the fully admitted Linux LLVM SDK");
  }
  const aliasStat = fs.lstatSync(alias), targetStat = regularSnapshot(target, "admitted llvm-ar", 256 << 20), targetHash = sourceIO.fileHash(target);
  if (!aliasStat.isSymbolicLink() || fs.readlinkSync(alias) !== catalog.toolchain.llvmLibTarget || fs.realpathSync(alias) !== target ||
      (targetStat.mode & 0o111n) === 0n || Number(targetStat.size) !== targetRow.bytes || targetHash !== targetRow.sha256) {
    throw new Error("Local AMD llvm-lib alias or admitted llvm-ar bytes changed");
  }
  fs.accessSync(target, fs.constants.X_OK);
  regularSnapshot(resultFile, "Linux LLVM SDK receipt", 8 << 20);
  return {path: alias, output: input.llvmOutput, resultSha256: sourceIO.fileHash(resultFile),
    identity: {relative: catalog.toolchain.llvmLibAlias, aliasTarget: catalog.toolchain.llvmLibTarget,
      bytes: Number(targetStat.size), sha256: targetHash}};
}

/** Apply the existing offline ROCm source verifier to the stable local workspace cohort.
 * @param {ReturnType<typeof localRocm.localRocmSdkPaths>} paths @param {NodeJS.ProcessEnv} env @param {string} git */
function verifyLocalAmdhipSources(paths, env, git) {
  for (const directory of [paths.source, path.join(paths.source, "roots/clr"), paths.pythonSourceRoot]) {
    const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Local AMD pinned source directory is missing or non-canonical: " + directory);
    }
  }
  for (const file of [paths.fetchReceipt, paths.rootsFile, paths.pythonArchive, paths.pythonSourceReceipt]) {
    regularSnapshot(file, "local ROCm source receipt/archive", 64 << 20);
  }
  return nativeCi.verifyRocmSources({env}, paths, git);
}

/** Build this tiny source-generated input from already-admitted local source and Linux LLVM SDKs.
 * No CI identity, fetch, build or output repair is performed.
 * @param {{sourceRoot:string,llvmOutput:string,llvmLib:string,output:string,localBuildId:string}} input
 * @param {NodeJS.ProcessEnv} [environment] @param {{verifySources?:typeof verifyLocalAmdhipSources,
 * verifyLlvm?:typeof verifyLinuxLlvmAdmission}} [dependencies] */
async function prepareLocalAmdhipWindows(input, environment = process.env, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows AMD import-library generation requires Linux x64 (WSL2)");}
  if (!environment || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Local Windows AMD generation requires MOM_AOT_BUILD_ONLY=1");}
  if (!sourceIO.hasExactBuildFields(input, ["llvmLib", "llvmOutput", "localBuildId", "output", "sourceRoot"])) {
    throw new Error("Local Windows AMD input has unknown or missing fields: " + (input && typeof input === "object" ? Object.keys(input).sort().join(",") : "not an object"));
  }
  if (![input.sourceRoot, input.llvmOutput, input.llvmLib, input.output].every(absolute)) {
    throw new Error("Local Windows AMD input paths must be canonical and absolute");
  }
  if (typeof input.localBuildId !== "string" || !localBuildIdPattern.test(input.localBuildId)) {
    throw new Error("Local Windows AMD input needs UUID localBuildId");
  }
  const selected = windowsAmdhipCatalog(), workspace = path.dirname(input.sourceRoot), sourcePaths = localRocm.localRocmSdkPaths(workspace),
    recipe = amdhipRecipe(), recipeSha256 = createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), producerSha256 = sourceIO.fileHash(__filename),
    git = "/usr/bin/git", outputStat = fs.lstatSync(input.output, {throwIfNoEntry: false});
  if (sourcePaths.source !== input.sourceRoot || outputStat || input.output.startsWith(repo + path.sep) ||
      path.dirname(input.output) !== fs.realpathSync(path.dirname(input.output)) || [input.sourceRoot, input.llvmOutput].some(root => sourceIO.buildPathsOverlap(root, input.output))) {
    throw new Error("Local AMD output must be new, canonical, outside the checkout, and disjoint from source and LLVM inputs");
  }
  for (const directory of [workspace, path.join(workspace, "sources"), input.sourceRoot, path.join(input.sourceRoot, "roots/clr"), input.llvmOutput]) {
    const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Local AMD source or LLVM input directory is missing or non-canonical: " + directory);
    }
  }
  const marker = path.join(workspace, ".mom-native-sdk-workspace.json"), markerStat = fs.lstatSync(marker, {throwIfNoEntry: false});
  if (!markerStat?.isFile() || markerStat.isSymbolicLink() || fs.realpathSync(marker) !== marker) {throw new Error("Local AMD source workspace is not marked for this checkout");}
  localSdk.admitWorkspace(workspace, repo);
  const verifySources = dependencies.verifySources || verifyLocalAmdhipSources, verifyLlvm = dependencies.verifyLlvm || verifyLinuxLlvmAdmission,
    env = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  const gitStat = regularSnapshot(git, "Git", 64 << 20);
  if ((gitStat.mode & 0o111n) === 0n) {throw new Error("Pinned source Git verifier is not executable");}
  verifySources(sourcePaths, env, git);
  const rootsBefore = sourceIO.fileHash(sourcePaths.rootsFile), sourceReceiptSha256 = sourceIO.fileHash(sourcePaths.fetchReceipt),
    tool = verifyLocalAmdhipTool(input, verifyLlvm), toolchain = {role: selected.toolchain.llvmRole,
      sdkOutput: tool.output, sdkResultSha256: tool.resultSha256, llvmLib: tool.identity},
    verifyInputs = () => {
      verifySources(sourcePaths, env, git);
      const currentTool = verifyLocalAmdhipTool(input, verifyLlvm);
      if (sourceIO.fileHash(sourcePaths.rootsFile) !== rootsBefore || sourceIO.fileHash(sourcePaths.fetchReceipt) !== sourceReceiptSha256 ||
          !isDeepStrictEqual(currentTool, tool) || sourceIO.fileHash(__filename) !== producerSha256 ||
          !isDeepStrictEqual(amdhipRecipe(), recipe) || !isDeepStrictEqual(windowsAmdhipCatalog(), selected)) {
        throw new Error("Local AMD source, tool, catalog or producer recipe changed during generation");
      }
    };
  return generateWindowsAmdhipPayload({selected, sourceRoot: input.sourceRoot, destination: input.output,
    stagingParent: path.dirname(input.output), stagingPrefix: ".native-windows-amdhip-local-" + input.localBuildId,
    identity: {localBuildId: input.localBuildId}, toolchain, toolPath: tool.path, fetchReceiptSha256: sourceReceiptSha256,
    rootsReceiptSha256: rootsBefore, recipe, recipeSha256, producerSha256, verifyInputs});
}

/** POSIX write bits are an admission invariant; Windows has no equivalent read-only directory semantics. @param {import("node:fs").BigIntStats} stat */
function writableMode(stat) {return process.platform !== "win32" && (stat.mode & 0o222n) !== 0n;}
/** @param {string} directory @returns {AmdhipReceipt} */
function admitWindowsAmdhip(directory) {
  const selected = windowsAmdhipCatalog();
  const licenseName = path.posix.basename(selected.source.license.path);
  if (!absolute(directory) || fs.realpathSync(directory) !== directory) {throw new Error("Select a canonical Windows AMD input directory");}
  const rootStat = fs.lstatSync(directory, {bigint: true});
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || writableMode(rootStat) ||
      !isDeepStrictEqual(fs.readdirSync(directory).sort(), [licenseName, selected.library.fileName, "sdk.json"].sort())) {
    throw new Error("Windows AMD input tree has an invalid or writable top-level layout");
  }
  const receiptFile = path.join(directory, "sdk.json"), receiptStat = regularSnapshot(receiptFile, "receipt", 1 << 20),
    receipt = /** @type {AmdhipReceipt} */ (readBuildJson(receiptFile)), recipe = amdhipRecipe(), source = receipt?.source,
    toolchain = receipt?.toolchain, def = receipt?.definition, library = receipt?.library, licenseProof = receipt?.files?.[licenseName],
    libraryProof = receipt?.files?.[selected.library.fileName];
  if (!libraryProof || !licenseProof) {throw new Error("Windows AMD receipt is stale, malformed or incomplete");}
  const localReceipt = Boolean(receipt && typeof receipt === "object" && Object.hasOwn(receipt, "localBuildId")),
    receiptKeys = ["version", "target", "status", "networked", "buildOnly", "compiled", "executed", "vendorPrebuilt",
      "sourceGenerated", "publicationQualified", "redistributionQualified", "producerSha256", "recipe", "source", "toolchain", "definition",
      "library", "output", "files", "note", ...localReceipt ? ["localBuildId"] : []],
    identityValid = !localReceipt || typeof receipt.localBuildId === "string" && localBuildIdPattern.test(receipt.localBuildId),
    ciSourceSuffix = path.basename(source?.sourceRoot || "").slice("mom-native-rocm-prerequisites-".length),
    sourceRootValid = localReceipt ? absolute(source?.sourceRoot) && path.basename(source.sourceRoot) === "rocm-sources" :
      validCiSuffix(ciSourceSuffix) && source?.sourceRoot === "/tmp/mom-native-rocm-prerequisites-" + ciSourceSuffix,
    toolchainIdentityValid = localReceipt ? sourceIO.hasExactBuildFields(toolchain, ["role", "sdkOutput", "sdkResultSha256", "llvmLib", "command"]) &&
      absolute(toolchain.sdkOutput) && /^[a-f0-9]{64}$/.test(toolchain.sdkResultSha256 || "") :
      sourceIO.hasExactBuildFields(toolchain, ["role", "handoffRoot", "handoffSha256", "llvmLib", "command"]) && provenancePath(toolchain.handoffRoot) &&
      /^[a-f0-9]{64}$/.test(toolchain.handoffSha256 || "");
  if (writableMode(receiptStat) || !sourceIO.hasExactBuildFields(receipt, receiptKeys) || receipt.version !== 2 || receipt.target !== "windows-amdhip64" || receipt.status !== "PASS" ||
      receipt.networked !== false || receipt.buildOnly !== true || receipt.compiled !== false || receipt.executed !== false || receipt.vendorPrebuilt !== false ||
      receipt.sourceGenerated !== true || receipt.publicationQualified !== false || receipt.redistributionQualified !== false ||
      !identityValid || !/^[a-f0-9]{64}$/.test(receipt.producerSha256 || "") ||
      !sourceIO.hasExactBuildFields(receipt.recipe, ["sha256", "files"]) || receipt.recipe.sha256 !== createHash("sha256").update(JSON.stringify(recipe)).digest("hex") ||
      !isDeepStrictEqual(receipt.recipe.files, recipe.graph) || !sourceIO.hasExactBuildFields(source, ["origin", "revision", "sourceRoot", "fetchReceiptSha256", "rootsReceiptSha256", "git", "definition", "license"]) ||
      source.origin !== selected.source.origin || source.revision !== selected.source.revision ||
      !sourceRootValid ||
      !/^[a-f0-9]{64}$/.test(source.fetchReceiptSha256 || "") || !/^[a-f0-9]{64}$/.test(source.rootsReceiptSha256 || "") ||
      !sourceIO.hasExactBuildFields(source.git, ["path", "bytes", "sha256"]) || source.git.path !== "/usr/bin/git" || !Number.isSafeInteger(source.git.bytes) ||
      source.git.bytes < 1 || !/^[a-f0-9]{64}$/.test(source.git.sha256 || "") ||
      !sourceIO.hasExactBuildFields(source.definition, ["path", "blob", "bytes", "sha256", "actualSha256"]) ||
      source.definition.path !== selected.source.definition.path || source.definition.blob !== selected.source.definition.blob ||
      source.definition.bytes !== selected.source.definition.bytes || source.definition.sha256 !== selected.source.definition.sha256 || source.definition.actualSha256 !== selected.source.definition.sha256 ||
      !sourceIO.hasExactBuildFields(source.license, ["path", "blob", "bytes", "sha256", "actualSha256", "note"]) ||
      source.license.path !== selected.source.license.path || source.license.blob !== selected.source.license.blob ||
      source.license.bytes !== selected.source.license.bytes || source.license.sha256 !== selected.source.license.sha256 || source.license.actualSha256 !== selected.source.license.sha256 ||
      typeof source.license.note !== "string" || source.license.note.length > 256 ||
      !toolchainIdentityValid || toolchain.role !== selected.toolchain.llvmRole ||
      !sourceIO.hasExactBuildFields(toolchain.llvmLib, ["relative", "aliasTarget", "bytes", "sha256"]) || toolchain.llvmLib.relative !== selected.toolchain.llvmLibAlias ||
      toolchain.llvmLib.aliasTarget !== selected.toolchain.llvmLibTarget || !Number.isSafeInteger(toolchain.llvmLib.bytes) || toolchain.llvmLib.bytes < 1 ||
      !/^[a-f0-9]{64}$/.test(toolchain.llvmLib.sha256 || "") ||
      !isDeepStrictEqual(toolchain.command, ["/def:<staging>/amdhip.def", "/machine:x64", "/out:<staging>/" + selected.library.fileName]) ||
      !sourceIO.hasExactBuildFields(def, ["path", "sourceBlob", "sourceSha256", "bytes", "derivedBytes", "derivedSha256", "transformation"]) ||
      def.path !== selected.source.definition.path || def.sourceBlob !== selected.source.definition.blob || def.sourceSha256 !== selected.source.definition.sha256 ||
      def.bytes !== selected.source.definition.bytes || def.derivedBytes !== selected.source.definition.derivedBytes ||
      def.derivedSha256 !== selected.source.definition.derivedSha256 || def.transformation !== "Only a LIBRARY directive is prepended to the unchanged upstream .def bytes." ||
      !sourceIO.hasExactBuildFields(library, ["fileName", "dllName", "bytes", "sha256"]) ||
      library.fileName !== selected.library.fileName || library.dllName !== selected.library.dllName || library.bytes !== selected.library.bytes ||
      library.sha256 !== selected.library.sha256 ||
      !provenancePath(receipt.output) || !sourceIO.hasExactBuildFields(receipt.files, [selected.library.fileName, licenseName]) ||
      !sourceIO.hasExactBuildFields(libraryProof, ["kind", "bytes", "sha256", "source"]) || libraryProof.kind !== "file" || libraryProof.bytes !== selected.library.bytes ||
      libraryProof.sha256 !== selected.library.sha256 || !sourceIO.hasExactBuildFields(libraryProof.source, ["kind", "path", "blob", "sha256"]) ||
      libraryProof.source.kind !== "generated-from-pinned-definition" || libraryProof.source.path !== selected.source.definition.path ||
      libraryProof.source.blob !== selected.source.definition.blob || libraryProof.source.sha256 !== selected.source.definition.sha256 ||
      !sourceIO.hasExactBuildFields(licenseProof, ["kind", "bytes", "sha256", "source"]) || licenseProof.kind !== "file" ||
      licenseProof.bytes !== selected.source.license.bytes || licenseProof.sha256 !== selected.source.license.sha256 ||
      !sourceIO.hasExactBuildFields(licenseProof.source, ["kind", "path", "blob"]) || licenseProof.source.kind !== "original-source-license-blob" ||
      licenseProof.source.path !== selected.source.license.path || licenseProof.source.blob !== selected.source.license.blob ||
      typeof receipt.note !== "string" || receipt.note.length > 512) {throw new Error("Windows AMD receipt is stale, malformed or incomplete");}
  const files = [{name: selected.library.fileName, bytes: selected.library.bytes, sha256: selected.library.sha256},
    {name: licenseName, bytes: selected.source.license.bytes, sha256: selected.source.license.sha256}];
  const snapshots = files.map(row => {
    const file = path.join(directory, row.name), result = verifyFile(file, row.name, row.bytes, row.sha256, row.bytes);
    if (writableMode(result.before)) {throw new Error("Windows AMD output file is writable: " + row.name);}
    return {file, before: result.before};
  });
  const catalogStat = regularSnapshot(catalogFile, "catalog", 32 << 10), catalogHash = sourceIO.fileHash(catalogFile);
  if (!sameSnapshot(receiptStat, regularSnapshot(receiptFile, "receipt", 1 << 20)) ||
      snapshots.some(row => !sameSnapshot(row.before, regularSnapshot(row.file, "payload", maxLibraryBytes))) ||
      !sameSnapshot(rootStat, fs.lstatSync(directory, {bigint: true})) || !sameSnapshot(catalogStat, regularSnapshot(catalogFile, "catalog", 32 << 10)) ||
      sourceIO.fileHash(catalogFile) !== catalogHash) {throw new Error("Windows AMD input changed during admission");}
  return receipt;
}

/** Consumer API retained: canonical paths and hashes for only the generated import library and its source-license evidence. @param {string} directory */
function windowsAmdhipInputs(directory) {
  const root = path.resolve(directory), receiptFile = path.join(root, "sdk.json"), receipt = admitWindowsAmdhip(root), receiptSha256 = sourceIO.fileHash(receiptFile),
    amdhip64 = path.join(root, catalog.library.fileName), license = path.join(root, path.posix.basename(catalog.source.license.path)), files = {
      [receiptFile]: receiptSha256, [amdhip64]: catalog.library.sha256, [license]: catalog.source.license.sha256, [catalogFile]: sourceIO.fileHash(catalogFile)};
  for (const [file, hash] of Object.entries(files)) {if (sourceIO.fileHash(file) !== hash) {throw new Error("Windows AMD input changed after admission: " + file);}}
  return {receipt, receiptFile, receiptSha256, amdhip64, license, files};
}

if (require.main === module) {
  (async () => {
    try {
      const [sourceRoot, toolBundleRoot, handoffSha256, output, extra] = process.argv.slice(2);
      if (!sourceRoot || !toolBundleRoot || !handoffSha256 || !output || extra) {
        throw new Error("Usage: prepare-native-amdhip-windows.js /tmp/mom-native-rocm-prerequisites-RUN-ATTEMPT /runner-temp/native-linux-llvm-tools-RUN-ATTEMPT <handoff-sha256> /new/native-windows-amdhip-RUN-ATTEMPT");
      }
      const receipt = await prepareWindowsAmdhip(sourceRoot, toolBundleRoot, handoffSha256, output);
      console.log("[native-sdk] Generated only the pinned AMD amdhip64.lib from the upstream CLR definition; no installer, library redistribution or legal conclusion.");
      console.log("[native-sdk] Output: " + receipt.output + " (" + receipt.library.bytes + " bytes, SHA-256 " + receipt.library.sha256 + ")");
    } catch (error) {console.error(error instanceof Error ? error.message : "Windows AMD import-library generation failed"); process.exitCode = 1;}
  })();
}

module.exports = {windowsAmdhipCatalog, prepareWindowsAmdhip, admitWindowsAmdhip, windowsAmdhipInputs, amdhipRecipe,
  deriveDefinition, verifyLinuxToolHandoff, prepareLocalAmdhipWindows};
