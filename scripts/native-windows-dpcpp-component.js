"use strict";

// Transport the verified original Windows runtime/JIT closure, not compiler installations or
// test executables. The existing adapter producer owns its library order and constructor roots.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {pins} = require("./probe-static-dpcpp"), {isStaticArchive} = require("./native-audit");
const {coffSectionPayloads} = require("./native-compiler-isolation");
const repo = path.resolve(__dirname, "..");
const wholeNames = ["mom-sycl-static.lib", "ze_loader.lib"], objectNames = ["drivers.obj", "driver-aliases.obj"];
const requiredNames = ["ur_loader.lib", "ur_common.lib", "ur_adapter_level_zero.lib", "ur_adapter_opencl.lib", "level_zero_utils.lib",
  "umf.lib", "umf_ba.lib", "umf_coarse.lib", "umf_utils.lib", "hwloc.lib", "OpenCL.lib", "cuda.lib", "level-zero-v2.lib", "sycl-jit.lib"];
/** @param {unknown} value @returns {value is string} */
const absolute = value => typeof value === "string" && path.isAbsolute(value) && value.length <= 4096 &&
  !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);
/** @typedef {{libraries: string[], wholeArchives: string[], objects: string[]}} WindowsDpcppLinks */

/** The original producer supplies membership; this guard only requires the complete feature
 * boundary and prevents basename collisions, traversal and executable/shared-library payloads.
 * @param {unknown} value @param {boolean} relative @returns {WindowsDpcppLinks} */
function windowsDpcppLinks(value, relative) {
  const r = /** @type {WindowsDpcppLinks} */ (value);
  if (!r || Object.keys(r).sort().join(",") !== "libraries,objects,wholeArchives" ||
      ![r.libraries, r.wholeArchives, r.objects].every(Array.isArray) || r.libraries.length < 64 || r.libraries.length > 512 ||
      r.wholeArchives.length !== 2 || r.objects.length !== 2) {throw new Error("Incomplete Windows DPC++ component link closure");}
  const rows = [...r.libraries, ...r.wholeArchives, ...r.objects];
  if (rows.some(n => typeof n !== "string" || (relative ? !/^(?:lib\/[A-Za-z0-9_.-]+\.lib|objects\/[A-Za-z0-9_.-]+\.obj)$/.test(n) : !absolute(n))) ||
      new Set(rows.map(n => path.basename(n).toLowerCase())).size !== rows.length ||
      requiredNames.some(n => !r.libraries.some(file => path.basename(file) === n)) ||
      !isDeepStrictEqual(r.wholeArchives.map(n => path.basename(n)), wholeNames) || !isDeepStrictEqual(r.objects.map(n => path.basename(n)), objectNames) ||
      [...r.libraries, ...r.wholeArchives].some(n => !n.endsWith(".lib") || relative && !n.startsWith("lib/")) ||
      r.objects.some(n => !n.endsWith(".obj") || relative && !n.startsWith("objects/"))) {throw new Error("Invalid Windows DPC++ component membership");}
  return r;
}

/** Rehash a bounded installed closure on any build host. Original source/tool paths are
 * provenance, not dependencies after export; this is not authentication of an untrusted SDK.
 * @param {string} root */
function windowsDpcppComponentInputs(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root) {throw new Error("Select a canonical Windows DPC++ component root");}
  const manifest = path.join(root, "component.json");
  if (!fs.lstatSync(manifest).isFile() || fs.realpathSync(manifest) !== manifest) {throw new Error("Windows component receipt must be a regular local file");}
  const sha256 = fileHash(manifest), r = readBuildJson(manifest);
  if (r.version !== 1 || r.kind !== "dpcpp-windows" || r.buildOnly !== true || r.publicationQualified !== false || !isDeepStrictEqual(r.pins, pins) || !r.files || typeof r.files !== "object" || Array.isArray(r.files)) {
    throw new Error("Invalid Windows DPC++ component receipt");
  }
  const link = windowsDpcppLinks(r.link, true), names = [...link.libraries, ...link.wholeArchives, ...link.objects];
  if (!isDeepStrictEqual(Object.keys(r.files).sort(), [...names].sort())) {throw new Error("Windows DPC++ component file membership differs");}
  const files = /** @type {Record<string, string>} */ ({[manifest]: sha256});
  for (const name of names) {
    const file = path.join(root, name), hash = r.files[name];
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file || typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {
      throw new Error("Windows DPC++ component bytes/type changed: " + name);
    }
    if (name.startsWith("lib/")) {if (!isStaticArchive(file)) {throw new Error("Windows DPC++ component needs static archives");}} else {coffSectionPayloads(file);}
    files[file] = hash;
  }
  for (const directory of ["lib", "objects"]) {
    if (!isDeepStrictEqual(fs.readdirSync(path.join(root, directory)).sort(), names.filter(n => n.startsWith(directory + "/")).map(n => path.basename(n)).sort())) {
      throw new Error("Unrecorded Windows DPC++ component file");
    }
  }
  if (fileHash(manifest) !== sha256) {throw new Error("Windows DPC++ component receipt changed during admission");}
  const resolve = (/** @type {string[]} */ rows) => rows.map(n => path.join(root, n));
  return {link: {libraries: resolve(link.libraries), wholeArchives: resolve(link.wholeArchives), objects: resolve(link.objects)}, files};
}

/** Consume optional relocatable SDK components without expanding their producer trees. Keep
 * explicit extras first, then the host closure, then the original runtime/JIT order. Never
 * deduplicate by basename: independent private LLVM runtimes may legitimately share names.
 * A CUDA image include is selection metadata here; its bytes are admitted by the application reader.
 * @param {WindowsDpcppLinks & {dpcppComponent?: string, hostComponent?: string, nvrtcSdk?: string, cudaImages?: string, includeRoots?: string[], nvmlHeader?: string, hipDriverLibrary?: string}} sdk */
function windowsDpcppSdkInputs(sdk) {
  const baseKeys = "libraries,objects,wholeArchives", applicationKeys = "hipDriverLibrary,includeRoots,libraries,nvmlHeader,objects,wholeArchives",
    keys = Object.keys(sdk || {}).sort(), shape = keys.filter(key => !["cudaImages", "dpcppComponent", "hostComponent", "nvrtcSdk"].includes(key)).join(",");
  if (!sdk || typeof sdk !== "object" || Array.isArray(sdk) ||
      ![baseKeys, applicationKeys].includes(shape)) {
    throw new Error("Invalid Windows SDK component selections");
  }
  if (Object.hasOwn(sdk, "dpcppComponent") && !absolute(sdk.dpcppComponent)) {throw new Error("Select an absolute Windows DPC++ component root");}
  if (Object.hasOwn(sdk, "hostComponent") && !absolute(sdk.hostComponent)) {throw new Error("Select an absolute Windows host component root");}
  if (Object.hasOwn(sdk, "nvrtcSdk") && !absolute(sdk.nvrtcSdk)) {throw new Error("Select an absolute Windows vendor NVRTC SDK root");}
  if (Object.hasOwn(sdk, "cudaImages") && (!absolute(sdk.cudaImages) || path.extname(sdk.cudaImages) !== ".inc")) {
    throw new Error("Select an absolute Windows CUDA image include ending in .inc");
  }
  if (Object.hasOwn(sdk, "cudaImages") && Object.hasOwn(sdk, "nvrtcSdk")) {
    throw new Error("Windows CUDA image include and vendor NVRTC SDK selections are mutually exclusive");
  }
  const component = Object.hasOwn(sdk, "dpcppComponent") ? windowsDpcppComponentInputs(/** @type {string} */ (sdk.dpcppComponent)) : null;
  // Load this independent reader lazily: its producer imports SDK planning code that can
  // indirectly depend on this consumer module during startup.
  const host = Object.hasOwn(sdk, "hostComponent") ? require("./native-windows-host-component").windowsHostComponentInputs(/** @type {string} */ (sdk.hostComponent)) : null;
  // The import archive is consumed only by driver discovery, never by the static link.
  // Do not permit a separate SDK path to bypass the host producer's admitted bytes.
  if (host && Object.hasOwn(sdk, "hipDriverLibrary") && sdk.hipDriverLibrary !== host.runtimeInputs["amdhip64"]) {
    throw new Error("Windows application HIP driver library must match the host component");
  }
  /** @type {Record<string, string>} */ let nvrtcFiles = {};
  if (Object.hasOwn(sdk, "nvrtcSdk")) {
    if (process.platform !== "linux") {throw new Error("Windows vendor NVRTC SDK provenance is Linux-bound; select it from a Linux cross-build host");}
    const root = /** @type {string} */ (sdk.nvrtcSdk), producer = require("./prepare-native-nvrtc-sdk"),
      admitted = producer.nvrtcSdkInputs("windows", root), receipt = admitted.receipt, plan = producer.memberPlan("windows"),
      libraryRows = plan.filter(row => "source" in row && row.output.startsWith("lib/x64/") && row.output.endsWith(".lib"));
    if (receipt.output !== root || admitted.receiptFile !== path.join(root, "sdk.json") || libraryRows.length !== 3) {
      throw new Error("Windows vendor NVRTC SDK changed or has an incomplete static library cohort");
    }
    for (const row of libraryRows) {
      const expected = path.join(root, ...row.output.split("/")), matches = sdk.libraries.filter(file => path.basename(file).toLowerCase() === path.basename(expected).toLowerCase());
      if (matches.length !== 1 || matches[0] !== expected) {throw new Error("Windows application must select each admitted vendor NVRTC archive exactly once: " + row.output);}
    }
    nvrtcFiles = admitted.files;
  }
  const link = {libraries: /** @type {string[]} */ ([]), wholeArchives: /** @type {string[]} */ ([]), objects: /** @type {string[]} */ ([])};
  for (const name of /** @type {const} */ (["libraries", "wholeArchives", "objects"])) {
    const maximum = name === "objects" ? 64 : 512, suffix = name === "objects" ? /\.(?:o|obj)$/i : /\.(?:a|lib)$/i;
    if (!Array.isArray(sdk[name]) || sdk[name].length > maximum || sdk[name].some(file => !absolute(file) || !suffix.test(file))) {throw new Error("Windows SDK links must be bounded literal static inputs");}
    link[name] = [...sdk[name], ...(name === "libraries" ? host?.link.libraries || [] : []), ...(component?.link[name] || [])];
    if (link[name].length > maximum) {throw new Error("Merged Windows SDK links exceed bounded input limits");}
  }
  const files = [...link.libraries, ...link.wholeArchives, ...link.objects];
  if (new Set(files.map(file => file.toLowerCase())).size !== files.length) {throw new Error("Duplicate Windows SDK/component input");}
  const hashes = /** @type {Record<string, string>} */ ({}), foldedHashes = new Set();
  for (const cohort of [host?.files, component?.files, nvrtcFiles]) {
    for (const [file, hash] of Object.entries(cohort || {})) {
      if (!absolute(file) || typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {throw new Error("Invalid Windows SDK component hash entry: " + file);}
      if (foldedHashes.has(file.toLowerCase())) {throw new Error("Duplicate Windows SDK component receipt/input: " + file);}
      foldedHashes.add(file.toLowerCase());
      hashes[file] = hash;
    }
  }
  return {link, files: hashes};
}

/** Stage only an already-admitted link fragment. Public for bounded structural fixtures;
 * real exports must first use exportWindowsDpcppComponent's original-source admission.
 * @param {WindowsDpcppLinks} selected @param {{path: string, sha256: string}} producer @param {string} output */
function stageWindowsDpcppComponent(selected, producer, output) {
  const links = windowsDpcppLinks(selected, false), originals = [...links.libraries, ...links.wholeArchives, ...links.objects];
  if (!absolute(output) || fs.lstatSync(output, {throwIfNoEntry: false}) || path.join(fs.realpathSync(path.dirname(output)), path.basename(output)) !== output) {
    throw new Error("Windows DPC++ component output must be new and canonical");
  }
  const nested = (/** @type {string} */ parent, /** @type {string} */ child) => {const n = path.relative(parent, child); return !n || !path.isAbsolute(n) && n !== ".." && !n.startsWith(".." + path.sep);};
  const inputs = /** @type {Record<string, string>} */ ({});
  for (const file of originals) {
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file || nested(output, file) || nested(path.dirname(file), output)) {throw new Error("Windows component overlaps or aliases a producer input");}
    inputs[file] = fileHash(file);
    if (file.endsWith(".lib")) {if (!isStaticArchive(file)) {throw new Error("Windows component input is not a static archive");}} else {coffSectionPayloads(file);}
  }
  fs.mkdirSync(output, {mode: 0o700}); fs.mkdirSync(path.join(output, "lib")); fs.mkdirSync(path.join(output, "objects"));
  const names = (/** @type {string[]} */ rows, /** @type {string} */ directory) => rows.map(n => directory + "/" + path.basename(n));
  const link = {libraries: names(links.libraries, "lib"), wholeArchives: names(links.wholeArchives, "lib"), objects: names(links.objects, "objects")};
  const files = /** @type {Record<string, string>} */ ({}), targetNames = [...link.libraries, ...link.wholeArchives, ...link.objects];
  for (const [index, file] of originals.entries()) {
    const name = targetNames[index]; if (!name) {throw new Error("Incomplete Windows component export plan");}
    const destination = path.join(output, name); fs.copyFileSync(file, destination, fs.constants.COPYFILE_EXCL);
    if (fileHash(destination) !== inputs[file] || fileHash(file) !== inputs[file]) {throw new Error("Windows component input changed during export");}
    files[name] = inputs[file] || "";
  }
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("Windows component input changed during export");}}
  fs.writeFileSync(path.join(output, "component.json"), JSON.stringify({version: 1, kind: "dpcpp-windows", pins, link, files, producer, buildOnly: true, publicationQualified: false}, null, 2) + "\n", {flag: "wx"});
  return windowsDpcppComponentInputs(output);
}

/** Verify the original SDK, JIT, adapter isolation and lazy bindings without copying them.
 * Fresh exports and local warm builds share this admission; no producer receipt is restamped.
 * @param {string} producer @param {string} cuda @param {NodeJS.ProcessEnv} [environment] */
function windowsDpcppProducerInputs(producer, cuda, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1" || ![producer, cuda].every(absolute)) {throw new Error("Windows component export requires Linux, explicit roots and MOM_AOT_BUILD_ONLY=1");}
  const producerSha = fileHash(producer), r = readBuildJson(producer);
  if (r.version !== 1 || r.buildOnly !== true || r.executed !== false || r.publicationQualified !== false || !r.recipe || !r.inputs || !r.headers) {throw new Error("Invalid Windows adapter producer receipt");}
  const {adapterSdkInputs, windowsRuntimeLinks} = require("./prepare-native-windows-adapters");
  const before = adapterSdkInputs(r.sdkFile, r.descriptorFile, cuda), bound = /** @type {Record<string, string>} */ ({...r.inputs, ...r.headers, [producer]: producerSha});
  if (before.sdk.sourceJit?.enabled !== true || !r.hostJitBinding || r.hostJitBinding.executed !== false) {throw new Error("Windows component requires complete source JIT");}
  verifySourceInputs(r.recipe, sourceInputs(repo, [], Object.keys(r.recipe.graph)), "Windows adapter export recipe");
  for (const row of [...r.adapters, ...r.driver.objects]) {
    const file = row.archive || row.path; if (!absolute(file) || fileHash(file) !== row.sha256) {throw new Error("Windows adapter/driver payload changed");} bound[file] = row.sha256;
  }
  for (const row of r.adapters) {
    const isolated = readBuildJson(row.isolation); bound[row.isolation] = fileHash(row.isolation);
    if (isolated.prefix !== "mom_ur_" + row.kind.replaceAll("-", "_") + "_private_" || !isDeepStrictEqual(isolated.entryPoints, [row.factory]) ||
        isolated.transformed?.length !== 1 || isolated.transformed[0]?.output !== row.archive || isolated.transformed[0]?.sha256 !== row.sha256 ||
        ["memberOrderPreserved", "nonDirectivePayloadsPreserved", "mappedDirectivesPreserved", "symbolOrderPreserved", "auxiliaryRecordsPreserved", "relocationsPreserved"].some(n => isolated.checks?.[n] !== true)) {
      throw new Error("Windows adapter isolation preservation differs");
    }
  }
  const links = windowsRuntimeLinks(before.sdk.artifacts, r.adapters, r.driver.objects, before.sdk.sourceJit.libraries);
  if (!isDeepStrictEqual(links, r.link)) {throw new Error("Windows component differs from its producer link selection");}
  // Bind the raw runtime and isolated compiler payloads too, not only their receipt files.
  for (const row of before.sdk.artifacts) {bound[row.path] = row.sha256;}
  const jit = require("./build-native-windows-sdk").windowsSourceJitInputs(before.sdk.sourceJit.receiptFile);
  if (!jit) {throw new Error("Missing Windows source-JIT payload hashes");} Object.assign(bound, jit.files);
  for (const file of [...links.libraries, ...links.wholeArchives, ...links.objects]) {bound[file] = fileHash(file);}
  for (const [file, hash] of Object.entries(bound)) {if (!absolute(file) || fileHash(file) !== hash) {throw new Error("Windows component provenance changed");}}
  return {links, producer: {path: producer, sha256: producerSha}, files: bound};
}

/** Join the installed closure to its admitted original producer, not just its self-authored hashes.
 * This also preserves archive order and rejects an otherwise-valid component from another build.
 * @param {string} output @param {ReturnType<typeof windowsDpcppProducerInputs>} original */
function matchWindowsDpcppComponent(output, original) {
  const result = windowsDpcppComponentInputs(output), manifest = readBuildJson(path.join(output, "component.json"));
  if (!isDeepStrictEqual(manifest.producer, original.producer)) {throw new Error("Windows component belongs to another original producer");}
  for (const kind of /** @type {const} */ (["libraries", "wholeArchives", "objects"])) {
    const selected = original.links[kind], installed = result.link[kind];
    if (selected.length !== installed.length || selected.some((file, index) => {
      const target = installed[index];
      return !target || path.basename(file) !== path.basename(target) || original.files[file] !== result.files[target];
    })) {throw new Error("Windows component differs from its original ordered payloads");}
  }
  for (const [file, hash] of Object.entries(original.files)) {if (fileHash(file) !== hash) {throw new Error("Windows component provenance changed during admission");}}
  return result;
}

/** Export a fresh component only after full source admission; warm users call the same matcher.
 * @param {string} producer @param {string} cuda @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function exportWindowsDpcppComponent(producer, cuda, output, environment = process.env) {
  if (!absolute(output)) {throw new Error("Windows component output must be absolute");}
  const original = windowsDpcppProducerInputs(producer, cuda, environment);
  stageWindowsDpcppComponent(original.links, original.producer, output);
  return matchWindowsDpcppComponent(output, original);
}

module.exports = {windowsDpcppLinks, windowsDpcppComponentInputs, windowsDpcppSdkInputs, stageWindowsDpcppComponent,
  windowsDpcppProducerInputs, matchWindowsDpcppComponent, exportWindowsDpcppComponent};
if (require.main === module) {
  try {
    const [producer, cuda, output, extra] = process.argv.slice(2);
    if (!producer || !cuda || !output || extra) {throw new Error("Usage: native-windows-dpcpp-component.js /adapters/result.json /CUDA /new/component");}
    const result = exportWindowsDpcppComponent(producer, cuda, output); console.log(JSON.stringify(result.link, null, 2));
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
