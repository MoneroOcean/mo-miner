"use strict";

// Native CI Linux source-JIT resources, SDK phases, runtime proof and the static DPC++ component.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readBuildJson, readBuildJsonSnapshot, readBuildText, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {auditGnuHeaders} = require("./prepare-native-gnu-headers");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {jitSdkPlan} = require("./native-jit-sdk");
const {jitBuildStages, jitBuildOutputs} = require("./native-jit-stages");
const {dpcppEntries, compilerArchives: jitCompilerArchives} = require("./native-compiler-isolation");
const {dpcppComponentInputs} = require("./native-dpcpp-component");
const {REQUIRED: jitBaseRequired, pinnedCommitDate} = require("./prepare-native-jit-base");
const {resourceTarget, verifyJitResources} = require("./prepare-native-sycl-jit-resources");
const {appendWorkflowOutputs, canonicalExecutable, hasExactKeys, runNode} = require("./native-ci-job");
const {extractState, listState, packState, safeRelative, temp} = require("./native-ci-tar");
const {dpcppCudaInputPins, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, gnuHeadersStatePaths, linuxSourceStatePaths,
  provisionOneApi, verifyLinuxSources} = require("./native-ci-linux-sources");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const repo = path.resolve(__dirname, "..");

/** Preserve every producer's original absolute path while keeping the generated JIT state separate.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function linuxJitResourcesStatePaths(paths, environment = process.env) {
  const linux = linuxSourceStatePaths(paths, environment), dpcpp = dpcppCudaInputsStatePaths(paths, environment),
    gnu = gnuHeadersStatePaths(paths, environment), name = "mom-native-linux-jit-resources-" + paths.suffix,
    root = path.join(temp, name), sdkTar = path.join(paths.runnerTemp, "native-linux-jit-resources-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_LINUX_JIT_RESOURCES_DIR"] !== root || !path.isAbsolute(paths.runnerTemp) ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native Linux JIT resource root must exactly match this workflow run");
  }
  return {linux, dpcpp, gnu, name, root, emitter: path.join(root, "emitter"), bootstrap: path.join(root, "bootstrap"),
    bootstrapBuild: path.join(root, "bootstrap/build"), bootstrapOverlay: path.join(root, "bootstrap-overlay"),
    bootstrapOverlayEvidence: path.join(root, "bootstrap-overlay.json"),
    base: path.join(root, "base"), profile: path.join(root, "resource-profile.json"), resources: path.join(root, "resources"),
    configureEvidence: path.join(root, "bootstrap-configure.json"), result: path.join(root, "ci-result.json"), sdkTar,
    incomingSourceTar: linux.incomingTar, incomingDpcppTar: path.join(paths.runnerTemp, "incoming", path.basename(dpcpp.sdkTar)),
    incomingGnuTar: path.join(paths.runnerTemp, "incoming", path.basename(gnu.sdkTar))};
}

/** @typedef {Pick<ReturnType<typeof linuxJitResourcesStatePaths>, "root" | "emitter" | "bootstrap" | "bootstrapBuild" |
 * "bootstrapOverlay" | "bootstrapOverlayEvidence" | "base" | "profile" | "resources" | "configureEvidence"> &
 * {linux:import("./native-ci-linux-sources").LinuxPreparedSources,dpcpp:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState,gnu:{root:string}}} LinuxJitResourceTreeState */

/** Pure invocation plan for the existing emitter, headers-only configure and resource producers.
 * @param {LinuxJitResourceTreeState} state @param {Record<string,string>} roots
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools @param {string} sourceDate */
function linuxJitResourceBuildPlan(state, roots, tools, sourceDate) {
  if (!/^[0-9]{8}$/.test(sourceDate)) {throw new Error("JIT bootstrap needs the pinned DPC++ compiler date");}
  const configure = [...jitSdkPlan("linux", roots, state.dpcpp.dpcppRoot, state.bootstrapOverlay, state.bootstrap, tools, 1).configure,
    "-DSYCL_COMPILER_VERSION=" + sourceDate];
  return {sourceDate, emitter: {script: "scripts/native-jit-sdk.js", args: ["emitter", state.dpcpp.dpcppRoot, state.emitter, path.join(state.gnu.root, "target-headers.json")]},
    configure, base: {script: "scripts/prepare-native-jit-base.js", args: ["--sdk", state.dpcpp.dpcppRoot, "--cuda", state.dpcpp.cudaRoot,
      "--build", state.bootstrapBuild, "--sources", state.linux.compilerPreparedReceipt,
      "--dpcpp-git", path.join(state.linux.source, "roots/dpcpp"), "--output", state.base]},
    resources: {script: "scripts/prepare-native-sycl-jit-resources.js", args: ["--profile", state.profile, "--output", state.resources]}};
}

/** Build the CMake probe overlay from real admitted target headers, never placeholder bytes.
 * Directory links keep the bootstrap small; the source/GNU receipts are independently rechecked.
 * @param {LinuxJitResourceTreeState} state @param {{headers:{root:string,files?:unknown[]},notices?:unknown[]}} gnu
 * @param {{root:string,revision:string}} opencl */
function linuxJitBootstrapOverlayPlan(state, gnu, opencl) {
  const rows = [
    {virtual: "include/mom-cxx", source: path.join(gnu.headers.root, "include/mom-cxx"), kind: "gnu"},
    {virtual: "include/mom-cxx-target", source: path.join(gnu.headers.root, "include/mom-cxx-target"), kind: "gnu"},
    {virtual: "include/mom-system-target", source: path.join(gnu.headers.root, "include/mom-system-target"), kind: "gnu"},
    {virtual: "include/mom-system", source: path.join(gnu.headers.root, "include/mom-system"), kind: "gnu"},
    {virtual: "include/CL", source: path.join(opencl.root, "CL"), kind: "opencl"},
  ];
  if (!/^[a-f0-9]{40}$/.test(opencl.revision) || rows.some(row => !path.isAbsolute(row.source) || path.resolve(row.source) !== row.source)) {
    throw new Error("JIT bootstrap overlay needs admitted GNU target roots and pinned OpenCL-Headers");
  }
  return rows.map(row => ({...row, link: path.join(state.bootstrapOverlay, ...row.virtual.split("/"))}));
}

/** Materialize only directory links to admitted GNU/OpenCL header trees and bind their sources.
 * @param {LinuxJitResourceTreeState} state @param {{headers:{root:string,files?:unknown[]},notices?:unknown[]}} gnu
 * @param {{root:string,revision:string}} opencl */
function seedLinuxJitBootstrapOverlay(state, gnu, opencl) {
  if (fs.lstatSync(state.bootstrapOverlay, {throwIfNoEntry: false}) || fs.lstatSync(state.bootstrapOverlayEvidence, {throwIfNoEntry: false})) {
    throw new Error("JIT bootstrap overlay refuses existing state");
  }
  fs.mkdirSync(path.join(state.bootstrapOverlay, "include"), {recursive: true, mode: 0o700});
  const rows = linuxJitBootstrapOverlayPlan(state, gnu, opencl);
  for (const row of rows) {
    const source = fs.realpathSync(row.source), stat = fs.lstatSync(row.source);
    if (!stat.isDirectory() || stat.isSymbolicLink() || source !== row.source ||
        row.kind === "opencl" && !fs.lstatSync(path.join(source, "cl.h"), {throwIfNoEntry: false})?.isFile()) {
      throw new Error("JIT bootstrap overlay source is missing or not canonical: " + row.virtual);
    }
    fs.symlinkSync(source, row.link, "dir");
    if (!fs.lstatSync(row.link).isSymbolicLink() || fs.realpathSync(row.link) !== source) {
      throw new Error("JIT bootstrap overlay link differs from its admitted source: " + row.virtual);
    }
  }
  const record = {version: 1, root: state.bootstrapOverlay,
    roots: rows.map(({virtual, source, link, kind}) => ({virtual, source, link, kind})),
    gnuReceipt: {path: path.join(state.gnu.root, "result.json"), sha256: fileHash(path.join(state.gnu.root, "result.json"))},
    compilerReceipt: {path: state.linux.compilerPreparedReceipt, sha256: fileHash(state.linux.compilerPreparedReceipt)},
    openclRevision: opencl.revision};
  fs.writeFileSync(state.bootstrapOverlayEvidence, JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return record;
}

/** @param {LinuxJitResourceTreeState} state @param {unknown} expected */
function auditLinuxJitBootstrapOverlay(state, expected) {
  const record = readBuildJson(state.bootstrapOverlayEvidence);
  if (!isDeepStrictEqual(record, expected) || record.root !== state.bootstrapOverlay || !Array.isArray(record.roots) ||
      record.roots.length !== 5 || record.gnuReceipt?.sha256 !== fileHash(record.gnuReceipt.path) ||
      record.compilerReceipt?.sha256 !== fileHash(record.compilerReceipt.path)) {
    throw new Error("JIT bootstrap overlay provenance changed");
  }
  for (const row of record.roots) {
    if (!fs.lstatSync(row.link).isSymbolicLink() || fs.readlinkSync(row.link) !== row.source ||
        fs.realpathSync(row.link) !== row.source || !fs.statSync(row.link).isDirectory()) {
      throw new Error("JIT bootstrap overlay link changed: " + row.virtual);
    }
  }
  return record;
}

/** Bind the public resource profile only to admitted emitter, GNU package and JIT-base records.
 * @param {LinuxJitResourceTreeState} state
 * @param {{fixture:string,compiler:string,targetHeaders:string[]}} emitter
 * @param {{headers:{root:string},notices:Array<{package:string,path:string,sha256:string}>}} gnu */
function linuxJitResourceProfile(state, emitter, gnu) {
  const target = resourceTarget("x86_64-unknown-linux-gnu"), base = path.join(state.base, "base");
  return {version: 1, target: target.target, fixture: emitter.fixture, clang: emitter.compiler,
    receipt: path.join(state.emitter, "result.json"), gnuRoot: gnu.headers.root, baseRoot: base,
    noticeRoot: gnu.headers.root, syclInclude: path.join(base, "include"),
    clangBuiltinInclude: path.join(base, "lib/clang/23/include"), baseManifest: path.join(state.base, "base-manifest.json"),
    headerRoots: target.roots.map(virtual => ({path: path.join(gnu.headers.root, ...virtual.split("/")), virtual})),
    required: Object.fromEntries(Object.entries(jitBaseRequired).map(([name, row]) => [name, {virtual: row.virtual, sha256: row.sha256}])),
    notices: gnu.notices.map(notice => ({source: path.relative(gnu.headers.root, notice.path).split(path.sep).join("/"),
      virtual: "usr/share/licenses/" + notice.package + "/copyright", sha256: notice.sha256}))};
}

/** Produce the JIT resource tree from admitted directories, independently of CI transport.
 * Callers provide installed tools and containment; this recipe never installs packages.
 * @param {LinuxJitResourceTreeState} state
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {string} [expectedDpcppManifestSha256] */
function buildLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256) {
  if (fs.lstatSync(state.root, {throwIfNoEntry: false})) {throw new Error("Linux JIT resource stage refuses an existing output");}
  const git = canonicalExecutable("/usr/bin/git"), paths = {env: environment};
  verifyLinuxSources(paths, state.linux, git);
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), expectedDpcppManifestSha256);
  const gnu = auditGnuHeaders(state.gnu),
    compiler = verifyPreparedSdkSources(state.linux.compilerPreparedReceipt, "compiler"),
    compilerRoots = Object.fromEntries(Object.entries(compiler.prepared).map(([name, component]) => [name, component.root]));
  fs.mkdirSync(state.root, {mode: 0o700});
  const bootstrapOverlay = seedLinuxJitBootstrapOverlay(state, gnu, compiler.prepared.openclHeaders),
    sourceDate = pinnedCommitDate(path.join(state.linux.source, "roots/dpcpp"), paths.env);
  if (!sourceDate) {throw new Error("Pinned DPC++ compiler date is missing");}
  const plan = linuxJitResourceBuildPlan(state, compilerRoots, tools, sourceDate);
  runNode(plan.emitter.script, plan.emitter.args, paths.env, 5 * 60 * 1000);
  const emitterReceipt = readBuildJson(path.join(state.emitter, "result.json"));
  const configureLog = path.join(state.root, "bootstrap-configure.log"), descriptor = fs.openSync(configureLog, "wx", 0o600);
  let configured;
  try {
    configured = spawnSync(plan.configure[0] || "", plan.configure.slice(1), {cwd: repo, env: paths.env, stdio: ["ignore", descriptor, descriptor], timeout: 5 * 60 * 1000});
  } finally {fs.closeSync(descriptor);}
  const configureRecord = {version: 1, command: plan.configure, sourceDate, bootstrapOverlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
    tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}])),
    status: configured.status, signal: configured.signal || null, error: configured.error?.message || null,
    log: {path: configureLog, sha256: fileHash(configureLog)}};
  fs.writeFileSync(state.configureEvidence, JSON.stringify(configureRecord, null, 2) + "\n", {flag: "wx", mode: 0o600});
  if (configured.error || configured.signal || configured.status !== 0) {throw new Error("Headers-only JIT CMake configure failed; see " + configureLog, {cause: configured.error});}
  auditLinuxJitBootstrapOverlay(state, bootstrapOverlay);
  for (const file of [path.join(state.bootstrapBuild, "CMakeCache.txt"), path.join(state.bootstrapBuild, "build.ninja")]) {
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file) {throw new Error("Headers-only JIT configure omitted a regular build record");}
  }
  runNode(plan.base.script, plan.base.args, paths.env, 10 * 60 * 1000);
  const profile = linuxJitResourceProfile(state, emitterReceipt, gnu);
  fs.writeFileSync(state.profile, JSON.stringify(profile, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode(plan.resources.script, plan.resources.args, paths.env, 8 * 60 * 1000);
  return auditLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256);
}

/** Bind resource-manifest base rows and required libraries to admitted producer records.
 * @param {unknown} manifest @param {unknown} required
 * @param {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} expectedBaseRows */
function assertLinuxJitResourceManifest(manifest, required, expectedBaseRows) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest) || !Array.isArray(expectedBaseRows)) {
    throw new Error("Linux JIT resource manifest metadata is malformed");
  }
  const requiredObject = required && typeof required === "object" && !Array.isArray(required) ? required : null,
    requiredRows = /** @type {Array<{virtual?:unknown,sha256?:unknown}>} */ (requiredObject ? Object.values(requiredObject) : []);
  if (!requiredObject || !hasExactBuildFields(requiredObject, ["libdevice", "spirv32", "spirv64"]) ||
      requiredRows.some(row => !row || typeof row !== "object" || Array.isArray(row) ||
        !expectedBaseRows.some(base => base.virtual === row.virtual && base.sha256 === row.sha256))) {
    throw new Error("Linux JIT required device libraries are not present in the admitted base payload");
  }
  const files = /** @type {{files?:unknown}} */ (manifest).files;
  if (!Array.isArray(files) || files.some(row => !row || typeof row !== "object" || Array.isArray(row) ||
      row.root !== "base" && row.root !== "gnu")) {
    throw new Error("Linux JIT resource manifest has malformed or unknown file roots");
  }
  const actualBase = /** @type {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} */ (
      files.filter(row => row.root === "base")),
    byVirtual = (/** @type {Array<{virtual:string}>} */ rows) => [...rows].sort((left, right) =>
      left.virtual < right.virtual ? -1 : left.virtual > right.virtual ? 1 : 0);
  if (!isDeepStrictEqual(/** @type {{required?:unknown}} */ (manifest).required, required) ||
      !isDeepStrictEqual(byVirtual(actualBase), byVirtual(expectedBaseRows))) {
    throw new Error("Linux JIT resource manifest differs from the admitted base payload and required device libraries");
  }
}

/** Re-admit a completed resource tree without running an emitter/compiler or restamping receipts.
 * Returned fingerprints let a local cache bind input/tool and output identities across invocations.
 * @param {LinuxJitResourceTreeState} state
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {string} [expectedDpcppManifestSha256] */
function auditLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256) {
  const baseRoot = path.join(state.base, "base"), fixture = path.join(state.emitter, "native-kawpow-source"),
    baseStat = fs.lstatSync(baseRoot, {throwIfNoEntry: false}), fixtureStat = fs.lstatSync(fixture, {throwIfNoEntry: false});
  if (!baseStat?.isDirectory() || baseStat.isSymbolicLink() || fs.realpathSync(baseRoot) !== baseRoot ||
      !fixtureStat?.isFile() || fixtureStat.isSymbolicLink() || fs.realpathSync(fixture) !== fixture) {
    throw new Error("JIT resource base and emitter must be canonical real directory/file outputs");
  }
  verifyLinuxSources({env: environment}, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), expectedDpcppManifestSha256);
  const gnu = auditGnuHeaders(state.gnu), compiler = verifyPreparedSdkSources(state.linux.compilerPreparedReceipt, "compiler"),
    sourceDate = pinnedCommitDate(path.join(state.linux.source, "roots/dpcpp"), environment),
    emitterFile = path.join(state.emitter, "result.json"), emitter = readBuildJson(emitterFile),
    baseFile = path.join(state.base, "result.json"), base = readBuildJson(baseFile), baseManifest = path.join(state.base, "base-manifest.json"),
    targetHeaders = readBuildJson(path.join(state.gnu.root, "target-headers.json")),
    resources = verifyJitResources(state.resources, "x86_64-unknown-linux-gnu"), manifest = readBuildJson(path.join(state.resources, "manifest.json"));
  if (!sourceDate || emitter.version !== 1 || emitter.fixture !== path.join(state.emitter, "native-kawpow-source") ||
      emitter.compiler !== path.join(state.dpcpp.dpcppRoot, "bin/clang++") || !Array.isArray(targetHeaders) ||
      !isDeepStrictEqual(emitter.targetHeaders, targetHeaders) || emitter.buildOnly !== true || emitter.executed !== false ||
      emitter.publicationQualified !== false || emitter.fixtureSha256 !== fileHash(emitter.fixture) ||
      emitter.compilerSha256 !== fileHash(emitter.compiler) || manifest.receiptSha256 !== fileHash(emitterFile) ||
      manifest.fixtureSha256 !== emitter.fixtureSha256 || manifest.compilerSha256 !== emitter.compilerSha256) {
    throw new Error("Public Linux JIT emitter did not retain the admitted target-header set or current inputs");
  }
  const baseInventory = require("./prepare-native-jit-base").baseManifest(baseRoot);
  if (base.version !== 1 || base.baseRoot !== path.join(state.base, "base") || base.baseManifest !== baseManifest ||
      base.baseManifestSha256 !== fileHash(baseManifest) || manifest.baseManifestSha256 !== base.baseManifestSha256 ||
      !isDeepStrictEqual(base.required, jitBaseRequired) || base.sourceReceipt?.path !== state.linux.compilerPreparedReceipt ||
      base.dpcpp?.sdkRoot !== state.dpcpp.dpcppRoot || base.dpcpp?.commitDate !== sourceDate ||
      base.cuda?.root !== state.dpcpp.cudaRoot || base.build?.root !== state.bootstrapBuild || base.buildOnly !== true ||
      base.compiled !== false || base.executed !== false || base.publicationQualified !== false ||
      !isDeepStrictEqual(readBuildJson(baseManifest), baseInventory)) {
    throw new Error("Public JIT base receipt does not bind the current inputs and payload");
  }
  const overlay = {version: 1, root: state.bootstrapOverlay,
      roots: linuxJitBootstrapOverlayPlan(state, gnu, compiler.prepared.openclHeaders),
      gnuReceipt: {path: path.join(state.gnu.root, "result.json"), sha256: fileHash(path.join(state.gnu.root, "result.json"))},
      compilerReceipt: {path: state.linux.compilerPreparedReceipt, sha256: fileHash(state.linux.compilerPreparedReceipt)},
      openclRevision: compiler.prepared.openclHeaders.revision},
    roots = Object.fromEntries(Object.entries(compiler.prepared).map(([name, component]) => [name, component.root])),
    plan = linuxJitResourceBuildPlan(state, roots, tools, sourceDate), log = path.join(state.root, "bootstrap-configure.log"),
    configure = {version: 1, command: plan.configure, sourceDate,
      bootstrapOverlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
      tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}])),
      status: 0, signal: null, error: null, log: {path: log, sha256: fileHash(log)}};
  auditLinuxJitBootstrapOverlay(state, overlay);
  const expectedProfile = linuxJitResourceProfile(state, emitter, gnu),
    /** @type {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} */
    expectedBase = baseInventory.files.map(row => ({...row, root: /** @type {"base"} */ ("base"),
      bytes: fs.statSync(path.join(baseRoot, row.source)).size}));
  assertLinuxJitResourceManifest(manifest, expectedProfile.required, expectedBase);
  if (!isDeepStrictEqual(readBuildJson(state.configureEvidence), configure) ||
      !isDeepStrictEqual(readBuildJson(state.profile), expectedProfile)) {
    throw new Error("Linux JIT resource configure/profile differs from admitted inputs and tools");
  }
  const files = [emitterFile, fixture, baseFile, baseManifest, state.profile, state.configureEvidence, state.bootstrapOverlayEvidence,
      path.join(state.bootstrapBuild, "CMakeCache.txt"), path.join(state.bootstrapBuild, "build.ninja"),
      state.linux.fetchReceipt, state.linux.preparedReceipt, state.linux.compilerPreparedReceipt, state.dpcpp.manifest,
      path.join(state.gnu.root, "result.json"), path.join(state.gnu.root, "copyrights.json"), log, __filename],
    fingerprints = {...resources.inputs, ...emitter.inputs, [emitter.fixture]: emitter.fixtureSha256,
      ...Object.fromEntries(Object.values(tools).map(file => [file, fileHash(file)]))};
  for (const file of files) {
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file) {throw new Error("JIT resource admission requires regular canonical records: " + file);}
    fingerprints[file] = fileHash(file);
  }
  for (const [file, hash] of Object.entries(fingerprints)) {
    if (!path.isAbsolute(file) || !/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("JIT resource input or output changed: " + file);}
  }
  return {resources, sourceDate, fingerprints};
}

/** Restore and re-admit the exact same-run inputs, then produce only the CPU JIT resource pack.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxJitResources(paths) {
  const state = linuxJitResourcesStatePaths(paths, paths.env);
  if ([state.root, state.sdkTar].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Linux JIT resource stage refuses an existing output");
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env);
  dpcppSourceArchivePin(state.incomingDpcppTar, externalDpcppPins.bundleSha256);
  const inputArchives = [
    {file: state.incomingSourceTar, root: state.linux.sourceName, role: "linux-sources"},
    {file: state.incomingDpcppTar, root: state.dpcpp.name, role: "dpcpp-cuda-inputs"},
    {file: state.incomingGnuTar, root: state.gnu.name, role: "gnu-headers"},
  ].map(input => {
    const stat = fs.lstatSync(input.file);
    if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Linux JIT resource stage requires all three same-run input tars");}
    return {...input, sha256: fileHash(input.file)};
  });
  for (const input of inputArchives) {extractState(input.file, temp, [input.root]);}
  verifyLinuxSources(paths, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins.manifestSha256);
  auditGnuHeaders(state.gnu);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-" + paths.suffix, 25 * 60 * 1000);
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    {resources, sourceDate} = buildLinuxJitResourceTree(state, tools, paths.env, externalDpcppPins.manifestSha256);
  if (inputArchives.some(input => fileHash(input.file) !== input.sha256)) {throw new Error("Same-run JIT input tar changed during production");}
  const baseManifest = path.join(state.base, "base-manifest.json"),
    baseReceiptFile = path.join(state.base, "result.json"), emitterReceiptFile = path.join(state.emitter, "result.json"),
    resourceManifest = path.join(state.resources, "manifest.json");
  const result = {version: 1, target: resources.target, buildOnly: true, executed: false, publicationQualified: false,
    scope: "CPU-only source-JIT resource preparation. It does not build the JIT SDK/compiler object/static runtime or qualify a device/runtime/publication.",
    inputArchives: inputArchives.map(({file, root, role, sha256}) => ({role, name: path.basename(file), root, sha256})),
    emitter: {receipt: emitterReceiptFile, sha256: fileHash(emitterReceiptFile)},
    bootstrap: {configure: state.configureEvidence, configureSha256: fileHash(state.configureEvidence), sourceDate,
      overlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
      cache: fileHash(path.join(state.bootstrapBuild, "CMakeCache.txt")), ninja: fileHash(path.join(state.bootstrapBuild, "build.ninja"))},
    base: {receipt: baseReceiptFile, sha256: fileHash(baseReceiptFile), manifest: baseManifest, manifestSha256: fileHash(baseManifest)},
    resources: {manifest: resourceManifest, manifestSha256: resources.manifestHash, files: readBuildJson(resourceManifest).files.length,
      notices: readBuildJson(resourceManifest).notices.length}, tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]))};
  fs.writeFileSync(state.result, JSON.stringify(result, null, 2) + "\n", {flag: "wx", mode: 0o600});
  packState(state.sdkTar, temp, [state.name]); listState(state.sdkTar, [state.name]);
  for (const input of inputArchives) {fs.unlinkSync(input.file);}
  console.log("[native-ci] Prepared and admitted the same-run Linux JIT resource pack; no JIT SDK, GPU, runtime, or publication qualification.");
}

/** Continue the resource handoff through the public source-JIT compiler phases.
 * Immutable upstream tars are independently restored at their original roots; only the
 * mutable compiler tree moves between jobs. @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function linuxJitSdkStatePaths(paths, environment = process.env) {
  const resources = linuxJitResourcesStatePaths(paths, environment), name = "mom-native-linux-jit-sdk-" + paths.suffix,
    root = path.join(temp, name);
  if (environment["MOM_NATIVE_LINUX_JIT_SDK_DIR"] !== root || !path.isAbsolute(paths.runnerTemp) ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native Linux source-JIT SDK root must exactly match this workflow run");
  }
  const phaseTar = (/** @type {string} */ phase) => path.join(paths.runnerTemp, "native-linux-jit-sdk-" + paths.suffix + "-" + linuxJitSdkArtifactPhase(phase) + ".tar"),
    phaseIncomingTar = (/** @type {string} */ phase) => path.join(paths.runnerTemp, "incoming", path.basename(phaseTar(phase)));
  return {resources, linux: resources.linux, dpcpp: resources.dpcpp, gnu: resources.gnu, name, root,
    output: path.join(root, "compiler-sdk"), isolation: path.join(root, "compiler-isolation"),
    isolatedObject: path.join(root, "compiler-isolation/isolated/compiler.o"), staticOutput: path.join(root, "static-dpcpp-sdk"),
    staticComponent: path.join(root, "static-dpcpp-sdk.component"), configFile: path.join(paths.runnerTemp, "native-linux-jit-sdk-" + paths.suffix + ".json"),
    ciInputs: path.join(root, "ci-inputs.json"), staticCiInputs: path.join(root, "static-ci-inputs.json"), phaseTar, phaseIncomingTar,
    inputTars: [
      {file: resources.linux.incomingTar, root: resources.linux.sourceName, role: "linux-sources"},
      {file: resources.incomingDpcppTar, root: resources.dpcpp.name, role: "dpcpp-cuda-inputs"},
      {file: resources.incomingGnuTar, root: resources.gnu.name, role: "gnu-headers"},
      {file: path.join(paths.runnerTemp, "incoming", path.basename(resources.sdkTar)), root: resources.name, role: "jit-resources"},
    ]};
}

/** @param {string} phase */
function linuxJitSdkArtifactPhase(phase) {
  if (phase === "configure") {return "configured";}
  if (phase === "static-dpcpp") {return "static-dpcpp";}
  if (jitBuildStages.includes(phase)) {return phase;}
  throw new Error("Unknown Linux JIT SDK handoff phase: " + phase);
}

/** A tar-level receipt ties unchanged upstream artifacts together. The resource producer
 * independently records the first three hashes; this re-admits rather than restamps them.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {unknown} resourceRecord
 * @param {Record<string,string>} archiveHashes */
function linuxJitSdkInputManifest(state, resourceRecord, archiveHashes) {
  if (!resourceRecord || typeof resourceRecord !== "object" || Array.isArray(resourceRecord) ||
      !Array.isArray(Reflect.get(resourceRecord, "inputArchives")) ||
      !hasExactKeys(archiveHashes, state.inputTars.map(row => row.file))) {
    throw new Error("Linux source-JIT SDK inputs omit the original same-run archive proof");
  }
  const archives = state.inputTars.map(row => {
      const sha256 = archiveHashes[row.file];
      if (!/^[a-f0-9]{64}$/.test(sha256 || "")) {throw new Error("Linux source-JIT SDK archive hash is malformed");}
      return {role: row.role, name: path.basename(row.file), root: row.root, sha256};
    }), original = Reflect.get(resourceRecord, "inputArchives");
  if (original.length !== 3 || original.some((/** @type {unknown} */ value, /** @type {number} */ index) =>
    !hasExactKeys(value, ["role", "name", "root", "sha256"]) || !isDeepStrictEqual(value, archives[index]))) {
    throw new Error("JIT resource receipt does not bind the exact three immutable input tars");
  }
  return {version: 1, run: state.resources.linux.sourceName.slice("mom-native-linux-source-prerequisites-".length), archives};
}

/** Pure plans use the existing producer CLIs and exact same-run artifact names.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools */
function linuxJitSdkBuildPlan(state, phase, tools) {
  const phases = ["configure", ...jitBuildStages, "static-dpcpp"], index = phases.indexOf(phase);
  if (index < 0 || !hasExactKeys(tools, ["cc", "cmake", "cxx", "ninja"]) ||
      Object.values(tools).some(file => typeof file !== "string" || !path.isAbsolute(file) || /[\r\n\0]/.test(file))) {
    throw new Error("Linux source-JIT CI needs an existing producer phase and four explicit tools");
  }
  const previous = index ? phases[index - 1] || "" : "", inputTars = state.inputTars.map(row => ({
    name: path.basename(row.file), root: row.root, role: row.role}));
  if (["configure", ...jitBuildStages].includes(phase)) {
    /** @type {import("./build-native-jit-sdk").Config} */
    const config = {version: 1, target: "linux", sources: state.linux.compilerPreparedReceipt, sdk: state.dpcpp.dpcppRoot,
      resources: path.join(state.resources.root, "resources"), tools, jobs: 4};
    return {phase, previous, previousArtifact: previous ? linuxJitSdkArtifactPhase(previous) : "",
      previousTar: previous ? state.phaseIncomingTar(previous) : "", inputTars, config, output: state.output,
      commands: [{script: "scripts/build-native-jit-sdk.js", args: [state.configFile, state.output, phase]}],
      artifact: {name: path.basename(state.phaseTar(phase)), file: state.phaseTar(phase)}};
  }
  return {phase, previous, previousArtifact: "link", previousTar: state.phaseIncomingTar("link"), inputTars,
    output: state.staticOutput, commands: [
      {script: "scripts/native-compiler-isolation.js", args: [path.join(state.output, "build"), state.isolation]},
      {script: "scripts/build-native-dpcpp-sdk.js", args: [state.linux.preparedReceipt, state.dpcpp.dpcppRoot, state.dpcpp.cudaRoot,
        state.isolatedObject, state.staticOutput, tools.cmake, tools.ninja, "4", tools.cc, tools.cxx]},
    ], artifact: {name: path.basename(state.phaseTar(phase)), file: state.phaseTar(phase)}};
}

/** Re-admit the configure receipt's two content hashes, not just its derived archive list.
 * @param {string} output @param {unknown} value */
function auditLinuxJitSdkConfigureArtifacts(output, value) {
  if (!path.isAbsolute(output) || !value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Linux source-JIT configure evidence is incomplete");
  }
  const receipt = /** @type {{configuredSha256?: string, commandsSha256?: string}} */ (value);
  for (const [name, key] of /** @type {Array<[string, "configuredSha256" | "commandsSha256"]>} */ (
    [["configured-tools.json", "configuredSha256"], ["commands.log", "commandsSha256"]])) {
    const file = path.join(output, name), stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file ||
        typeof receipt[key] !== "string" || !/^[a-f0-9]{64}$/.test(receipt[key] || "") || fileHash(file) !== receipt[key]) {
      throw new Error("Linux source-JIT configure artifact hash changed: " + name);
    }
  }
}

/** Restore and validate unchanged dependency and prior-phase archives before tool provisioning.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase */
function restoreLinuxJitSdkInputs(paths, state, phase) {
  const phases = ["configure", ...jitBuildStages, "static-dpcpp"], phaseIndex = phases.indexOf(phase);
  if (phaseIndex < 0 || fs.lstatSync(state.root, {throwIfNoEntry: false}) ||
      state.inputTars.some(row => fs.lstatSync(path.join(temp, row.root), {throwIfNoEntry: false}))) {
    throw new Error("Linux source-JIT CI requires new run-scoped roots for this phase");
  }
  for (const row of state.inputTars) {
    const stat = fs.lstatSync(row.file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(row.file) !== row.file) {
      throw new Error("Linux source-JIT CI requires every named same-run dependency tar");
    }
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env), dpcppTar = state.inputTars.find(row => row.role === "dpcpp-cuda-inputs");
  if (!dpcppTar) {throw new Error("Linux source-JIT CI is missing its DPC++/CUDA input archive");}
  dpcppSourceArchivePin(dpcppTar.file, externalDpcppPins.bundleSha256);
  const resourceTar = state.inputTars[3]?.file || "";
  listState(resourceTar, [state.resources.name]); extractState(resourceTar, temp, [state.resources.name]);
  const resourceRecord = readBuildJson(state.resources.result),
    hashes = Object.fromEntries(state.inputTars.map(row => [row.file, fileHash(row.file)])),
    manifest = linuxJitSdkInputManifest(state, resourceRecord, hashes);
  for (const row of state.inputTars.slice(0, 3)) {
    if (row === dpcppTar) {dpcppSourceArchivePin(row.file, externalDpcppPins.bundleSha256);}
    listState(row.file, [row.root]); extractState(row.file, temp, [row.root]);
  }
  dpcppSourceArchivePin(dpcppTar.file, externalDpcppPins.bundleSha256);
  verifyLinuxSources(paths, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins.manifestSha256); auditGnuHeaders(state.gnu);
  verifyJitResources(state.resources.resources, "x86_64-unknown-linux-gnu");
  let previous = null;
  if (phaseIndex) {
    previous = phases[phaseIndex - 1] || "";
    const priorTar = state.phaseIncomingTar(previous);
    if (!previous) {throw new Error("Linux source-JIT CI phase ancestry is incomplete");}
    listState(priorTar, [state.name]); extractState(priorTar, temp, [state.name]);
    if (!isDeepStrictEqual(readBuildJson(state.ciInputs), manifest)) {
      throw new Error("Linux source-JIT same-run input archive hashes changed between phases");
    }
  } else {
    fs.mkdirSync(state.root, {mode: 0o700});
    fs.writeFileSync(state.ciInputs, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  }
  // Prior-phase receipts hash compiler files; callers provision before deep admission.
  return {manifest, hashes, previous};
}

/** Re-admit current source-JIT phase records and the exact original archive closure.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase */
function auditLinuxJitSdkState(state, phase) {
  const expectedConfig = linuxJitSdkBuildPlan(state, "configure", {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja",
    cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"}).config;
  if (!expectedConfig) {throw new Error("Linux source-JIT configure plan has no configuration");}
  return auditLinuxJitCompilerState(state.output, expectedConfig, phase);
}

/** Read-only phase admission shared by local builds and CI; transport identities stay with callers.
 * @param {string} output @param {import("./build-native-jit-sdk").Config} expectedConfig @param {string} phase */
function auditLinuxJitCompilerState(output, expectedConfig, phase) {
  if (expectedConfig.target !== "linux" || !["configure", ...jitBuildStages].includes(phase) || !path.isAbsolute(output)) {
    throw new Error("Linux JIT compiler admission requires an absolute output and a supported Linux phase");
  }
  const build = path.join(output, "build"), beforeFile = path.join(output, "before.json"),
    before = readBuildJson(beforeFile), configure = readBuildJson(path.join(output, "stage-configure.json")),
    current = phase === "configure" ? configure : readBuildJson(path.join(output, "stage-" + phase + ".json")),
    libraries = jitCompilerArchives(readBuildText(path.join(output, "commands.log")), build), outputs = jitBuildOutputs(build);
  verifySourceInputs(before.recipe, require("./build-native-jit-sdk").jitSdkRecipe(), "Linux JIT compiler SDK admission");
  auditLinuxJitSdkConfigureArtifacts(output, configure);
  const compilerRoots = Object.fromEntries(Object.entries(readBuildJson(expectedConfig.sources).prepared)
      .map(([name, source]) => [name, source.root])), resources = verifyJitResources(expectedConfig.resources, "x86_64-unknown-linux-gnu"),
    expectedPlan = jitSdkPlan("linux", compilerRoots, expectedConfig.sdk,
      resources.overlay, output, expectedConfig.tools, expectedConfig.jobs);
  if (before.version !== 1 || before.config?.target !== "linux" || before.plan?.target !== "linux" || before.plan.build !== build ||
      !isDeepStrictEqual(before.config, expectedConfig) || !isDeepStrictEqual(before.plan, expectedPlan) ||
      configure.version !== 1 || configure.stage !== "configure" || configure.beforeSha256 !== fileHash(beforeFile) ||
      configure.buildOnly !== true || !isDeepStrictEqual(before.supportLibraries, []) || !isDeepStrictEqual(configure.supportLibraries, []) ||
      !isDeepStrictEqual(configure.libraries, libraries) ||
      current.version !== 1 || current.stage !== phase || current.beforeSha256 !== fileHash(beforeFile) ||
      current.buildOnly !== true || !isDeepStrictEqual(current.supportLibraries, []) ||
      !isDeepStrictEqual(current.libraries, libraries) || !isDeepStrictEqual(current.outputs, outputs) ||
      !isDeepStrictEqual(current.inputs, buildFileInputs(Object.keys(current.inputs || {})))) {
    throw new Error("Linux source-JIT compiler phase receipt or archive graph changed");
  }
  if (phase !== "configure") {
    const phases = ["configure", ...jitBuildStages], index = phases.indexOf(phase), previous = phases[index - 1] || "",
      previousFile = path.join(output, "stage-" + previous + ".json"), {value: prior, sha256} = readBuildJsonSnapshot(previousFile);
    if (!previous || current.previousSha256 !== sha256 || current.configuredSha256 !== configure.configuredSha256 ||
        current.commandsSha256 !== configure.commandsSha256 || !isDeepStrictEqual(current.libraries, prior.libraries)) {
      throw new Error("Linux source-JIT phase ancestry changed");
    }
  }
  if (phase === "link") {
    const result = readBuildJson(path.join(output, "result.json")), proof = path.join(build, "bin/native-dpcpp-jit"),
      expectedLibraries = libraries.map(file => ({path: file, sha256: fileHash(file)}));
    if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" || result.proof?.path !== proof ||
        result.proof.sha256 !== fileHash(proof) || result.compiled !== true || result.buildOnly !== true || result.executed !== false ||
        result.publicationQualified !== false || result.sourcesFile !== before.config.sources || result.resources !== before.config.resources ||
        !isDeepStrictEqual(result.plan, before.plan) || !isDeepStrictEqual(result.recipe, before.recipe) ||
        !isDeepStrictEqual(result.inputs, current.inputs) || !isDeepStrictEqual(result.supportLibraries, current.supportLibraries) ||
        !isDeepStrictEqual(result.libraries, expectedLibraries)) {
      throw new Error("Linux source-JIT final producer result is incomplete or changed");
    }
  }
  return current;
}

/** Build one existing public source-JIT phase and transfer only its mutable build tree.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {string} phase */
function buildLinuxJitSdkPhase(paths, phase) {
  const state = linuxJitSdkStatePaths(paths, paths.env), {hashes, previous} = restoreLinuxJitSdkInputs(paths, state, phase);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-sdk-" + paths.suffix, 25 * 60 * 1000);
  if (previous) {auditLinuxJitSdkState(state, previous);}
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    plan = linuxJitSdkBuildPlan(state, phase, tools);
  const config = require("./build-native-jit-sdk").jitSdkConfig(plan.config);
  fs.writeFileSync(state.configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const command = plan.commands[0];
  if (!command) {throw new Error("Linux source-JIT phase has no public producer command");}
  runNode(command.script, command.args, paths.env, 50 * 60 * 1000);
  if (["configure", ...jitBuildStages].includes(phase) &&
      !isDeepStrictEqual(readBuildJson(path.join(state.output, "before.json")).config, plan.config)) {
    throw new Error("Public source-JIT producer changed its run-scoped configuration");
  }
  auditLinuxJitSdkState(state, phase);
  for (const [file, sha256] of Object.entries(hashes)) {if (fileHash(file) !== sha256) {throw new Error("Linux source-JIT dependency tar changed during production");}}
  packState(plan.artifact.file, temp, [state.name]); listState(plan.artifact.file, [state.name]);
  console.log("[native-ci] Handed off source-JIT phase " + phase + "; build-only, no GPU/runtime/publication qualification.");
}

/** Run-scoped, proof-only handoff kept separate from the mutable SDK phase tar.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths */
function linuxJitRuntimeProofPaths(paths) {
  const name = "mom-native-linux-jit-runtime-proof-" + paths.suffix,
    archive = path.join(paths.runnerTemp, "native-linux-jit-runtime-proof-" + paths.suffix + ".tar");
  return {name, root: path.join(temp, name), archive,
    incomingArchive: path.join(paths.runnerTemp, "incoming", path.basename(archive)),
    proof: path.join(temp, name, "native-dpcpp-jit"), receipt: path.join(temp, name, "result.json")};
}

/** Package only the linked executable and its original build-only result receipt.
 * @param {string} sourceProof @param {string} sourceReceipt @param {string} base @param {string} name @param {string} archive */
function packLinuxJitRuntimeProofFiles(sourceProof, sourceReceipt, base, name, archive) {
  if (!path.isAbsolute(sourceProof) || !path.isAbsolute(sourceReceipt) || !path.isAbsolute(base) ||
      !safeRelative(name) || name.includes("/") || !path.isAbsolute(archive)) {
    throw new Error("Linux source-JIT runtime proof paths are invalid");
  }
  const proofStat = fs.lstatSync(sourceProof, {throwIfNoEntry: false}), resultStat = fs.lstatSync(sourceReceipt, {throwIfNoEntry: false});
  if (!proofStat?.isFile() || proofStat.isSymbolicLink() || !(proofStat.mode & 0o111) ||
      proofStat.size > 512 * 1024 * 1024 || !resultStat?.isFile() || resultStat.isSymbolicLink() ||
      resultStat.size > 16 * 1024 * 1024) {
    throw new Error("Linux source-JIT runtime proof or original receipt is not a regular executable/file");
  }
  const proofSha256 = fileHash(sourceProof), {value: result, sha256: resultSha256} = readBuildJsonSnapshot(sourceReceipt);
  if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" ||
      result.proof?.path !== sourceProof || result.proof.sha256 !== proofSha256 || result.compiled !== true ||
      result.buildOnly !== true || result.executed !== false || result.publicationQualified !== false) {
    throw new Error("Linux source-JIT runtime proof is not bound to an unchanged build-only receipt");
  }
  const root = fs.realpathSync(base), packageRoot = path.join(root, name);
  if (fs.existsSync(packageRoot)) {throw new Error("Linux source-JIT runtime proof output root already exists");}
  fs.mkdirSync(packageRoot, {mode: 0o700});
  let archiveCreated = false;
  try {
    const proof = path.join(packageRoot, "native-dpcpp-jit"), receipt = path.join(packageRoot, "result.json");
    fs.copyFileSync(sourceProof, proof, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(proof, proofStat.mode & 0o777);
    fs.copyFileSync(sourceReceipt, receipt, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(receipt, resultStat.mode & 0o777);
    if (fileHash(proof) !== proofSha256 || fileHash(receipt) !== resultSha256) {
      throw new Error("Linux source-JIT runtime proof copy differs from its producer bytes");
    }
    const packed = packState(archive, root, [name]); archiveCreated = true;
    const members = listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort(),
      expected = [name, name + "/native-dpcpp-jit", name + "/result.json"].sort();
    if (!isDeepStrictEqual(members, expected)) {throw new Error("Linux source-JIT runtime proof tar has unexpected members");}
    return {proofSha256, resultSha256, bundleSha256: fileHash(archive), bundleBytes: packed.bytes};
  } catch (error) {
    if (archiveCreated) {fs.rmSync(archive, {force: true});}
    throw error;
  } finally {
    fs.rmSync(packageRoot, {recursive: true, force: true});
  }
}

/** Admit an exact same-run proof-only tar before giving its path to the runtime test.
 * @param {string} archive @param {string} base @param {string} name @param {string} originalProofPath
 * @param {{proofSha256: string, resultSha256: string, bundleSha256: string, bundleBytes: string}} expected */
function extractLinuxJitRuntimeProof(archive, base, name, originalProofPath, expected) {
  const hashes = [expected.proofSha256, expected.resultSha256, expected.bundleSha256];
  if (!path.isAbsolute(archive) || !path.isAbsolute(base) || !path.isAbsolute(originalProofPath) ||
      !safeRelative(name) || name.includes("/") || hashes.some(value => !/^[a-f0-9]{64}$/.test(value)) ||
      !/^[1-9][0-9]{0,8}$/.test(expected.bundleBytes) || Number(expected.bundleBytes) > 513 * 1024 * 1024) {
    throw new Error("Linux source-JIT runtime proof expectations are invalid");
  }
  const archiveStat = fs.lstatSync(archive, {throwIfNoEntry: false});
  if (!archiveStat?.isFile() || archiveStat.isSymbolicLink() || archiveStat.size !== Number(expected.bundleBytes) ||
      fileHash(archive) !== expected.bundleSha256) {
    throw new Error("Linux source-JIT runtime proof bundle bytes or hash changed");
  }
  const members = listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort(),
    wanted = [name, name + "/native-dpcpp-jit", name + "/result.json"].sort();
  if (!isDeepStrictEqual(members, wanted)) {throw new Error("Linux source-JIT runtime proof tar is missing or has unexpected members");}
  extractState(archive, base, [name]);
  const root = path.join(fs.realpathSync(base), name), proof = path.join(root, "native-dpcpp-jit"), receipt = path.join(root, "result.json"),
    entries = fs.readdirSync(root).sort(), proofStat = fs.lstatSync(proof), resultStat = fs.lstatSync(receipt);
  const message = "Linux source-JIT runtime proof files differ from the same-run outputs";
  if (!isDeepStrictEqual(entries, ["native-dpcpp-jit", "result.json"]) || !proofStat.isFile() || proofStat.isSymbolicLink() ||
      !(proofStat.mode & 0o111) || proofStat.nlink !== 1 || !resultStat.isFile() || resultStat.isSymbolicLink() || resultStat.nlink !== 1 ||
      fileHash(proof) !== expected.proofSha256) {throw new Error(message);}
  const {value: result} = readBuildJsonSnapshot(receipt, undefined, {sha256: expected.resultSha256, message});
  if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" ||
      result.proof?.path !== originalProofPath || result.proof.sha256 !== expected.proofSha256 || result.compiled !== true ||
      result.buildOnly !== true || result.executed !== false || result.publicationQualified !== false) {
    throw new Error("Linux source-JIT runtime proof lost its original build-only receipt");
  }
  return {root, proof, receipt, proofSha256: expected.proofSha256, resultSha256: expected.resultSha256};
}

/** Export the linked executable only after the full public producer admission succeeds.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function packLinuxJitRuntimeProof(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), transfer = linuxJitRuntimeProofPaths(paths);
  auditLinuxJitSdkState(state, "link");
  const output = path.join(state.output, "build/bin/native-dpcpp-jit"), result = path.join(state.output, "result.json"),
    packed = packLinuxJitRuntimeProofFiles(output, result, temp, transfer.name, transfer.archive);
  appendWorkflowOutputs(paths.env, {proof_sha256: packed.proofSha256, result_sha256: packed.resultSha256,
    bundle_sha256: packed.bundleSha256, bundle_bytes: String(packed.bundleBytes)});
  console.log("[native-ci] Exported only the same-run source-JIT proof and unchanged build-only receipt; no SDK, GPU, or publication handoff.");
}

/** Validate/extract the separately uploaded proof tar; the compiler SDK is not an input.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function admitLinuxJitRuntimeProof(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), transfer = linuxJitRuntimeProofPaths(paths), env = paths.env,
    expected = {proofSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_PROOF_SHA256"] || "",
      resultSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_RESULT_SHA256"] || "",
      bundleSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_BUNDLE_SHA256"] || "",
      bundleBytes: env["MOM_NATIVE_LINUX_JIT_RUNTIME_BUNDLE_BYTES"] || ""};
  const admitted = extractLinuxJitRuntimeProof(transfer.incomingArchive, temp, transfer.name,
    path.join(state.output, "build/bin/native-dpcpp-jit"), expected);
  appendWorkflowOutputs(env, {fixture_path: admitted.proof, proof_sha256: admitted.proofSha256});
  console.log("[native-ci] Admitted the exact same-run proof-only tar and unchanged receipt; no SDK/compiler inputs restored.");
}

/** Isolate the completed original compiler closure and invoke the existing full static
 * DPC++ producer. Neither step executes a GPU or qualifies runtime/publication.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxJitStaticSdk(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), {manifest, hashes, previous} = restoreLinuxJitSdkInputs(paths, state, "static-dpcpp");
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-sdk-" + paths.suffix, 25 * 60 * 1000);
  if (previous) {auditLinuxJitSdkState(state, previous);}
  const linked = readBuildJson(path.join(state.output, "result.json"));
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    plan = linuxJitSdkBuildPlan(state, "static-dpcpp", tools);
  if (fs.lstatSync(state.isolation, {throwIfNoEntry: false}) || linked.target !== "linux" || linked.compiled !== true ||
      linked.executed !== false || linked.publicationQualified !== false) {
    throw new Error("Linux source-JIT isolation requires an admitted final link tree and new output");
  }
  const isolate = plan.commands[0], build = plan.commands[1];
  if (!isolate || !build) {throw new Error("Static DPC++ handoff omitted an existing public producer command");}
  runNode(isolate.script, isolate.args, paths.env, 15 * 60 * 1000);
  const receipt = readBuildJson(path.join(state.isolation, "result.json")), raw = path.join(state.isolation, "compiler-raw.o");
  if (receipt.input !== raw || receipt.object !== state.isolatedObject || receipt.prefix !== "mom_dpcpp23_" ||
      !isDeepStrictEqual(receipt.entryPoints, dpcppEntries) || receipt.inputSha256 !== fileHash(raw) ||
      receipt.objectSha256 !== fileHash(state.isolatedObject)) {throw new Error("Linux compiler isolation receipt does not bind the linked object");}
  if (fs.lstatSync(state.staticOutput, {throwIfNoEntry: false}) || fs.lstatSync(state.staticComponent, {throwIfNoEntry: false}) ||
      fs.lstatSync(state.staticCiInputs, {throwIfNoEntry: false})) {throw new Error("Linux static DPC++ output refuses existing evidence");}
  const staticInputs = {version: 1, run: manifest.run, sourceJitTar: {name: path.basename(state.phaseTar("link")), sha256: fileHash(state.phaseIncomingTar("link"))},
    isolatedObject: {path: state.isolatedObject, sha256: fileHash(state.isolatedObject)}, dependencies: manifest.archives};
  fs.writeFileSync(state.staticCiInputs, JSON.stringify(staticInputs, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode(build.script, build.args, paths.env, 50 * 60 * 1000);
  const produced = readBuildJson(path.join(state.staticOutput, "result.json")), component = dpcppComponentInputs(state.staticComponent);
  if (produced.version !== 1 || produced.target !== "linux" || produced.buildOnly !== true || produced.executed !== false ||
      produced.publicationQualified !== false || produced.archive !== path.join(state.staticOutput, "build/libmom-sycl-static.a") ||
      fileHash(produced.archive) !== produced.sha256 || component.files[path.join(state.staticComponent, "component.json")] !==
      fileHash(path.join(state.staticComponent, "component.json")) || !isDeepStrictEqual(readBuildJson(state.ciInputs), manifest) ||
      !isDeepStrictEqual(readBuildJson(state.staticCiInputs), staticInputs)) {
    throw new Error("Original static DPC++ producer did not preserve a complete admitted build-only component");
  }
  for (const [file, sha256] of Object.entries(hashes)) {if (fileHash(file) !== sha256) {throw new Error("Linux static DPC++ input tar changed during production");}}
  packState(plan.artifact.file, temp, [state.name]); listState(plan.artifact.file, [state.name]);
  console.log("[native-ci] Built and admitted the original static Linux DPC++ component; not executed or runtime/publication qualified.");
}

module.exports = {admitLinuxJitRuntimeProof, assertLinuxJitResourceManifest, auditLinuxJitBootstrapOverlay,
  auditLinuxJitCompilerState, auditLinuxJitResourceTree, auditLinuxJitSdkConfigureArtifacts, auditLinuxJitSdkState,
  buildLinuxJitResourceTree, buildLinuxJitResources, buildLinuxJitSdkPhase, buildLinuxJitStaticSdk, extractLinuxJitRuntimeProof,
  linuxJitBootstrapOverlayPlan, linuxJitResourceBuildPlan, linuxJitResourceProfile, linuxJitResourcesStatePaths,
  linuxJitRuntimeProofPaths, linuxJitSdkArtifactPhase, linuxJitSdkBuildPlan, linuxJitSdkInputManifest, linuxJitSdkStatePaths,
  packLinuxJitRuntimeProof, packLinuxJitRuntimeProofFiles, restoreLinuxJitSdkInputs, seedLinuxJitBootstrapOverlay};
