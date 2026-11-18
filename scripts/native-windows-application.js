"use strict";

// Full-application admission shared by Windows planning and final source freshness checks.
// This module compiles nothing, loads no GPU driver, and never accepts a partial factory set.
const fs = require("node:fs"), path = require("node:path");
const {absolutePath} = require("./build-native-release");
const {fileHash, readBuildJson, nativeSourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {nativeApplicationGpu} = require("./probe-native-controller");
const {isStaticArchive, auditNativeSymbols} = require("./native-audit");
const {coffSectionPayloads} = require("./native-compiler-isolation");
const {windowsDpcppSdkInputs} = require("./native-windows-dpcpp-component");
const {windowsGpuHandoffInputs} = require("./native-windows-gpu-handoff");
const nativeCudaImages = require("./native-cuda-images");
const {admittedJitCacheTag} = require("./native-jit-cache-tag");
const {assertUniqueHcfIds} = require("./native-hcf");
const {StringDecoder} = require("node:string_decoder"), {createHash} = require("node:crypto");

/** @typedef {{core: string, gpuReceipts: string[], defaultCompiler: string,
 * gpuHandoffs?: Array<{profile: string, runAttempt: string, receiptSha256: string, configSha256: string, handoffSha256: string}>,
 * tools: {llvmObjcopy: string, llvmAr: string},
 * sdk: {libraries: string[], wholeArchives: string[], objects: string[], includeRoots: string[], nvmlHeader: string, hipDriverLibrary: string, dpcppComponent?: string, hostComponent?: string, nvrtcSdk?: string, cudaImages?: string}}} WindowsApplication */

const embeddedCudaProfiles = new Set(["dpcpp", "acpp-cuda"]);
const embeddedCudaSources = ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"];
const staticNvrtcArchiveStems = new Set(["nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static"]);
const staticNvrtcInputs = new Set(["nvrtc.h", "nvptxcompiler.h", "nvrtc-headers.inc"]);
/** @typedef {{compiled?: Array<{file?: string, flags?: unknown}>,
 * sdkInputs?: Record<string, string>, profile?: string}} CudaImageGraphEvidence */

/** @param {string} file */
function isStaticNvrtcArchive(file) {
  const basename = path.basename(file).toLowerCase(), stem = basename.replace(/^lib/, "").replace(/\.(?:a|lib)$/i, "");
  return staticNvrtcArchiveStems.has(stem);
}

/** Require each maintained CUDA factory to prove it compiled the selected image cohort,
 * instead of trusting only the application-level descriptor. @param {CudaImageGraphEvidence} graph
 * @param {Record<string, string>} imageFiles */
function requireEmbeddedCudaReceipt(graph, imageFiles) {
  if (!Array.isArray(graph.compiled) || graph.compiled.length > 256 || !graph.sdkInputs ||
      typeof graph.sdkInputs !== "object" || Array.isArray(graph.sdkInputs)) {
    throw new Error("Embedded CUDA factory receipt lacks compiled-image evidence");
  }
  for (const source of embeddedCudaSources) {
    const rows = graph.compiled.filter((row) => row && row.file === source);
    const row = rows[0];
    if (rows.length !== 1 || !row || !Array.isArray(row.flags) ||
        !row.flags.includes("-DMOM_EMBEDDED_CUDA_IMAGES") ||
        row.flags.some((/** @type {unknown} */ flag) => typeof flag !== "string" ||
          flag.startsWith("-DMOM_STATIC_NVRTC") || flag.startsWith("-DMOM_EMBEDDED_NVRTC_HEADERS") ||
          /nvrtc-headers\.inc/i.test(flag))) {
      throw new Error(`Embedded CUDA factory ${graph.profile} did not compile ${source} with the image-only route`);
    }
  }
  for (const [file, hash] of Object.entries(imageFiles)) {
    if (graph.sdkInputs[file] !== hash) {throw new Error(`Embedded CUDA factory ${graph.profile} did not consume the selected image cohort: ${file}`);}
  }
  for (const input of Object.keys(graph.sdkInputs)) {
    if (staticNvrtcInputs.has(path.basename(input).toLowerCase()) || isStaticNvrtcArchive(input)) {
      throw new Error(`Embedded CUDA factory ${graph.profile} has a static NVRTC dependency: ${input}`);
    }
  }
}

/** @param {string[]} archives */
function rejectStaticNvrtcArchives(archives) {
  if (archives.some(isStaticNvrtcArchive)) {
    throw new Error("Embedded CUDA images cannot be mixed with static NVRTC libraries");
  }
}

/** @param {CudaImageGraphEvidence} graph */
function usesEmbeddedCudaImages(graph) {
  return Array.isArray(graph.compiled) && graph.compiled.some((row) =>
    Array.isArray(row?.flags) && row.flags.includes("-DMOM_EMBEDDED_CUDA_IMAGES"));
}

/** @param {unknown} value @param {string} label @param {number} maximum @param {boolean} [empty] @returns {string[]} */
function paths(value, label, maximum, empty = false) {
  if (!Array.isArray(value) || (!empty && !value.length) || value.length > maximum || value.some(item => !absolutePath(item)) || new Set(value).size !== value.length) {throw new Error("Invalid Windows application " + label);}
  return value;
}

/** Validate only explicit, literal build inputs before creating output or invoking tools.
 * @param {unknown} value @param {string} root @returns {WindowsApplication} */
function validateWindowsApplication(value, root) {
  const app = /** @type {WindowsApplication} */ (value);
  const appKeys = app && typeof app === "object" ? Object.keys(app).sort().join(",") : "",
    directKeys = "core,defaultCompiler,gpuReceipts,sdk,tools", handoffKeys = "core,defaultCompiler,gpuHandoffs,gpuReceipts,sdk,tools";
  if (!app || typeof app !== "object" || ![directKeys, handoffKeys].includes(appKeys) ||
      !absolutePath(app.core) || !/\.lib$/i.test(app.core)) {throw new Error("Invalid Windows application Core/descriptor");}
  paths(app.gpuReceipts, "GPU receipts", 64);
  const policy = nativeApplicationGpu(fs.readFileSync(path.join(root, "runtime/gpu-dispatch.cpp"), "utf8"), {defaultCompiler: app.defaultCompiler, flags: [], links: []});
  if (Object.hasOwn(app, "gpuHandoffs")) {
    const profiles = new Set(policy.factories.map(factory => factory.slice("mom_native_gpu_".length).replaceAll("_", "-")));
    if (!Array.isArray(app.gpuHandoffs) || app.gpuHandoffs.length !== profiles.size || app.gpuHandoffs.some(pin =>
      !pin || typeof pin !== "object" || Array.isArray(pin) || Object.keys(pin).sort().join(",") !== "configSha256,handoffSha256,profile,receiptSha256,runAttempt" ||
      typeof pin.profile !== "string" || !profiles.has(pin.profile) || typeof pin.runAttempt !== "string" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(pin.runAttempt) ||
      !/^[a-f0-9]{64}$/.test(pin.receiptSha256 || "") || !/^[a-f0-9]{64}$/.test(pin.configSha256 || "") || !/^[a-f0-9]{64}$/.test(pin.handoffSha256 || "")) ||
      new Set(app.gpuHandoffs.map(pin => pin.profile)).size !== profiles.size || new Set(app.gpuHandoffs.map(pin => pin.runAttempt)).size !== 1) {
      throw new Error("Windows application GPU handoffs must pin every maintained profile exactly once");
    }
  }
  if (!app.tools || Object.keys(app.tools).sort().join(",") !== "llvmAr,llvmObjcopy" || Object.values(app.tools).some(file => !absolutePath(file))) {throw new Error("Invalid Windows application LLVM tools");}
  const sdk = app.sdk;
  const sdkKeys = Object.keys(sdk || {}).sort(), sdkShape = sdkKeys.filter(key => !["dpcppComponent", "hostComponent", "nvrtcSdk", "cudaImages"].includes(key)).join(",");
  if (!sdk || sdkShape !== "hipDriverLibrary,includeRoots,libraries,nvmlHeader,objects,wholeArchives" ||
      Object.hasOwn(sdk, "dpcppComponent") && !absolutePath(sdk.dpcppComponent) ||
      Object.hasOwn(sdk, "hostComponent") && !absolutePath(sdk.hostComponent) ||
      Object.hasOwn(sdk, "nvrtcSdk") && !absolutePath(sdk.nvrtcSdk) ||
      Object.hasOwn(sdk, "cudaImages") && (!absolutePath(sdk.cudaImages) || path.extname(sdk.cudaImages) !== ".inc") ||
      Object.hasOwn(sdk, "cudaImages") && Object.hasOwn(sdk, "nvrtcSdk") ||
      !absolutePath(sdk.nvmlHeader) || path.basename(sdk.nvmlHeader).toLowerCase() !== "nvml.h" ||
      !absolutePath(sdk.hipDriverLibrary) || !/\.lib$/i.test(sdk.hipDriverLibrary)) {throw new Error("Invalid Windows application SDK");}
  for (const name of ["libraries", "wholeArchives"]) {
    if (paths(Reflect.get(sdk, name), name, 512, Object.hasOwn(sdk, "dpcppComponent")).some(file => !/\.(?:lib|a)$/i.test(file))) {throw new Error("Windows application SDK requires static archives");}
  }
  if (paths(sdk.objects, "SDK objects", 64, true).some(file => !/\.(?:obj|o)$/i.test(file))) {throw new Error("Windows application SDK requires COFF objects");}
  paths(sdk.includeRoots, "SDK include roots", 32);
  const libraries = [...sdk.libraries, ...sdk.wholeArchives, sdk.hipDriverLibrary];
  if (new Set(libraries).size !== libraries.length) {throw new Error("Duplicate Windows application SDK archive");}
  return app;
}

/** Verify a complete immutable file cohort. Hash receipts too, so replacing metadata during
 * compilation cannot silently substitute a second build. @param {Record<string, string>} files */
function verifyWindowsApplicationFiles(files) {
  for (const [file, hash] of Object.entries(files)) {if (fileHash(file) !== hash) {throw new Error("Windows application input changed: " + file);}}
}

/** Quote literal linker response-file tokens, not cmd.exe commands. In particular, ordinary
 * Windows path separators stay literal; only quotes and a trailing backslash run need escape.
 * @param {string[]} args */
function windowsResponseFile(args) {
  if (!Array.isArray(args) || args.length > 8192 || args.some(arg => typeof arg !== "string" || arg.length > 16384 || /[\0\r\n]/.test(arg))) {throw new Error("Invalid Windows linker response tokens");}
  return args.map(arg => '"' + arg.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/, "$1$1") + '"').join("\r\n") + "\r\n";
}

/** Full static GPU/compiler links have large text maps, not large individual symbol lines.
 * Audit a stable regular file with bounded storage instead of reading hundreds of MiB into
 * the controller heap. Never truncate inspection or skip symbols after a size threshold.
 * @param {string} file @param {RegExp} forbidden @param {string[]} [required] */
function auditWindowsLinkMap(file, forbidden, required = []) {
  if (!Array.isArray(required) || required.length > 32 || new Set(required).size !== required.length ||
      required.some(name => typeof name !== "string" || !/^[A-Za-z0-9_?@$%.]{1,4096}$/.test(name))) {throw new Error("Invalid required Windows link symbols");}
  const missing = new Set(required);
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || before.size < 1n || before.size > 512n * 1024n * 1024n) {throw new Error("Windows link map is not a bounded regular file");}
    const chunk = Buffer.allocUnsafe(64 << 10), decoder = new StringDecoder("utf8"), hash = createHash("sha256");
    const pattern = new RegExp(forbidden.source, forbidden.flags.replace(/[gy]/g, ""));
    let pending = "", bytes = 0n, lines = 0, maxLineBytes = 0;
    /** @param {string} line */
    const inspect = line => {
      const length = Buffer.byteLength(line);
      if (length > 1 << 20) {throw new Error("Windows link map symbol line exceeds bounds");}
      auditNativeSymbols(line);
      // LLD's map retains MSVC mangling. Match outer namespace roots only; SYCL's
      // nested graph::node types are legitimate and must not be mistaken for Node.js.
      if (/@(?:node|v8)@@/.test(line)) {throw new Error("Unexpected Node/V8/N-API symbols in Windows link map");}
      if (pattern.test(line)) {throw new Error("Windows link map contains a forbidden shim or intrinsic helper");}
      // Match a real public-symbol row, never map commentary, literals or a private prefix.
      const linked = missing.size ? line.match(/^\s*[0-9a-f]{4}:[0-9a-f]{8,16}\s+(\S+)\s+[0-9a-f]{16}(?:\s|$)/i)?.[1] : undefined;
      if (linked) {missing.delete(linked);}
      maxLineBytes = Math.max(maxLineBytes, length); lines++;
    };
    for (;;) {
      const size = fs.readSync(fd, chunk, 0, chunk.length, null); if (!size) {break;}
      bytes += BigInt(size); if (bytes > before.size) {throw new Error("Windows link map grew during audit");}
      hash.update(chunk.subarray(0, size)); pending += decoder.write(chunk.subarray(0, size));
      let start = 0, end;
      while ((end = pending.indexOf("\n", start)) >= 0) {inspect(pending.slice(start, end)); start = end + 1;}
      pending = pending.slice(start);
      if (Buffer.byteLength(pending) > 1 << 20) {throw new Error("Windows link map symbol line exceeds bounds");}
    }
    pending += decoder.end(); if (pending) {inspect(pending);}
    const after = fs.fstatSync(fd, {bigint: true});
    if (bytes !== before.size || before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) {throw new Error("Windows link map changed during audit");}
    if (missing.size) {throw new Error("Windows link omits required adapter symbols: " + [...missing].join(", "));}
    return {bytes: Number(bytes), sha256: hash.digest("hex"), lines, maxLineBytes};
  } finally {fs.closeSync(fd);}
}

/** Admit current-source Core and all original Windows GPU object graphs. Guest-side absolute
 * paths in a Core receipt are historical provenance; the exported sibling archive is matched
 * by content, never by restamping the receipt after transfer between build hosts.
 * @param {WindowsApplication} value @param {string} root */
function windowsApplicationInputs(value, root) {
  const app = validateWindowsApplication(value, root), {cudaImages: imagePath, ...sdkDescriptor} = app.sdk,
    selected = windowsDpcppSdkInputs(sdkDescriptor), sdk = selected.link;
  const embeddedImages = imagePath ? nativeCudaImages.cudaImageInputs(imagePath, root) : null;
  if (embeddedImages) {
    rejectStaticNvrtcArchives([...sdk.libraries, ...sdk.wholeArchives]);
  }
  const files = /** @type {Record<string, string>} */ ({...selected.files});
  for (const [file, hash] of Object.entries(embeddedImages?.files || {})) {
    if (Object.hasOwn(files, file) && files[file] !== hash) {throw new Error("Windows CUDA image input conflicts with an admitted SDK component: " + file);}
    files[file] = hash;
  }
  /** @param {string} file @param {string} [expected] */
  const admit = (file, expected) => {
    const hash = fileHash(file);
    if (expected !== undefined && (!/^[0-9a-f]{64}$/.test(expected) || hash !== expected)) {throw new Error("Windows application component hash differs: " + file);}
    files[file] = hash;
  };
  const coreFile = path.join(path.dirname(app.core), "inputs.json"), core = readBuildJson(coreFile); admit(coreFile);
  verifySourceInputs(core.sourceState, nativeSourceInputs(root, "cpu"), "Windows application Core");
  if (core.target !== "win32" || core.gpu !== true || core.portable !== true || core.archive?.stubs !== false ||
      !isStaticArchive(app.core) || core.archive?.sha256 !== core.objectSha256 || !Number.isSafeInteger(core.archive?.coffTableCount) || core.archive.coffTableCount < 1 ||
      core.archive.coffTableCount !== core.archive.machineCount || core.archive.machineCount !== core.archive.x64MachineCount) {throw new Error("Windows application requires a complete portable x64 Core archive");}
  admit(app.core, core.objectSha256);
  const policy = nativeApplicationGpu(fs.readFileSync(path.join(root, "runtime/gpu-dispatch.cpp"), "utf8"), {defaultCompiler: app.defaultCompiler, flags: [], links: []});
  const remaining = new Set(policy.factories), current = nativeSourceInputs(root, "gpu"), objects = [], graphs = [], hcfObjects = [], embeddedProfiles = new Set(),
    gpuHandoffMode = Object.hasOwn(app, "gpuHandoffs"), gpuHandoffPins = app.gpuHandoffs || [];
  if (app.gpuReceipts.length !== remaining.size) {throw new Error("Windows application requires every maintained GPU profile");}
  for (const file of app.gpuReceipts) {
    const receiptSha256 = gpuHandoffMode ? fileHash(file) : "", matchingPins = gpuHandoffMode ? gpuHandoffPins.filter(pin => pin.receiptSha256 === receiptSha256) : [],
      pin = matchingPins[0], handoff = pin ? windowsGpuHandoffInputs(file, pin) : null,
      graph = handoff?.graph || (gpuHandoffMode ? null : readBuildJson(file));
    if (gpuHandoffMode && !handoff) {throw new Error("Windows application GPU receipt has no matching pinned handoff");}
    if (gpuHandoffMode && matchingPins.length !== 1) {throw new Error("Windows application GPU receipt does not have exactly one handoff pin");}
    for (const [member, hash] of Object.entries(handoff?.files || {})) {
      if (Object.hasOwn(files, member) && files[member] !== hash) {throw new Error("Windows GPU handoff conflicts with an admitted application input: " + member);}
      files[member] = hash;
    }
    admit(file);
    if (!remaining.delete(graph.factory) || graph.factory !== "mom_native_gpu_" + String(graph.profile).replaceAll("-", "_") ||
        !absolutePath(graph.isolation) || !Array.isArray(graph.objects) || !graph.objects.length || graph.objects.length > 128) {throw new Error("Invalid or duplicate Windows application GPU factory");}
    verifySourceInputs(graph.sourceState, current, "Windows application GPU");
    const isolated = readBuildJson(graph.isolation); admit(graph.isolation);
    if (isolated.prefix !== "mom_" + graph.profile.replaceAll("-", "_") + "_gpu_" ||
        JSON.stringify(isolated.entryPoints) !== JSON.stringify([graph.factory]) || JSON.stringify(isolated.outputs) !== JSON.stringify(graph.objects) ||
        ["nonDirectivePayloadsPreserved", "nonDirectiveMetadataPreserved", "comdatSelectionPreserved", "associativeXdataPreserved", "linkerDirectiveReferencesPreserved", "externalUndefinedImportsPreserved", "privateUndefinedReferencesRenamed"]
          .some(key => isolated.checks?.[key] !== true) || isolated.checks?.oldCollisionDefinitions !== 0 || isolated.checks?.oldCollisionReferences !== 0) {throw new Error("Windows GPU isolation receipt differs from selected graph");}
    for (const row of graph.objects) {
      if (!absolutePath(row.path) || !/\.obj$/i.test(row.path)) {throw new Error("Invalid Windows GPU object path");}
      coffSectionPayloads(row.path); admit(row.path, row.sha256); objects.push(row.path);
      if (graph.profile.startsWith("acpp-")) {hcfObjects.push(row.path);}
    }
    if (!graph.sdkInputs || typeof graph.sdkInputs !== "object" || Array.isArray(graph.sdkInputs) || Object.keys(graph.sdkInputs).length > 16384 || !absolutePath(graph.configFile)) {throw new Error("Missing Windows GPU SDK/recipe inputs");}
    for (const [input, hash] of Object.entries(graph.sdkInputs)) {
      if (!absolutePath(input) || !/^[a-f0-9]{64}$/.test(hash || "")) {throw new Error("Invalid Windows GPU SDK input");}
      if (!handoff) {admit(input, hash);}
    }
    if (embeddedCudaProfiles.has(graph.profile)) {
      if (embeddedImages) {
        requireEmbeddedCudaReceipt(graph, embeddedImages.files);
        embeddedProfiles.add(graph.profile);
      } else if (usesEmbeddedCudaImages(graph)) {
        throw new Error("Embedded CUDA factory receipts require the matching sdk.cudaImages cohort");
      }
    }
    admit(graph.configFile, graph.configSha256);
    graphs.push({factory: graph.factory, receipt: file, receiptSha256: files[file]});
  }
  if (embeddedImages && [...embeddedCudaProfiles].some(profile => !embeddedProfiles.has(profile))) {
    throw new Error("Windows application requires embedded CUDA images in both dpcpp and acpp-cuda factories");
  }
  const allObjects = [...objects, ...sdk.objects];
  if (hcfObjects.length) {assertUniqueHcfIds(hcfObjects);}
  if (new Set(allObjects.map(file => fs.realpathSync(file))).size !== allObjects.length) {throw new Error("Duplicate Windows application/SDK object");}
  const libraries = [...sdk.libraries, ...sdk.wholeArchives, app.sdk.hipDriverLibrary];
  if (new Set(libraries.map(file => file.toLowerCase())).size !== libraries.length) {throw new Error("Duplicate Windows application SDK archive");}
  for (const file of libraries) {
    if (!isStaticArchive(file)) {throw new Error("Windows application SDK library is not a regular archive");} admit(file, files[file]);
  }
  for (const file of sdk.objects) {coffSectionPayloads(file);}
  for (const file of [...sdk.objects, app.sdk.nvmlHeader, ...Object.values(app.tools)]) {admit(file, files[file]);}
  if (embeddedImages) {verifyWindowsApplicationFiles(embeddedImages.files);}
  const jitCacheTag = admittedJitCacheTag(root, "windows-x64", {library: sdk.libraries,
    whole: sdk.wholeArchives, driver: [app.sdk.hipDriverLibrary], sdk: sdk.objects, gpu: objects});
  return {core: app.core, objects, graphs, files, jitCacheTag, definitions: policy.definitions};
}

module.exports = {validateWindowsApplication, windowsApplicationInputs, verifyWindowsApplicationFiles, rejectStaticNvrtcArchives, windowsResponseFile, auditWindowsLinkMap};
