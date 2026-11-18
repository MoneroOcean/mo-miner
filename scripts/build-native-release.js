"use strict";

// Linux native source producer. Reuse the maintained components without executing the result;
// release qualification and Engineering's direct-Node execution remain separate.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, sourceInputs, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {nativeApplicationGpu, nativeApplicationInputs, verifyCudaImageReceiptInputs} = require("./build-native-application");
const {releaseBuildEnvironment} = require("./build-environment");
const {nativeDependencies, auditNativeGlibc, isStaticArchive} = require("./native-audit");
const {nativeElf, verifyNativeStrip, elfReport, binutilsIdentity, readelfTool, stripTool} = require("./native-elf");
const {nativeIoInputs} = require("./build-native-io");
const {dpcppComponentInputs} = require("./native-dpcpp-component");
const staticNvrtcArchiveNames = new Set(["libnvrtc_static.a", "libnvrtc-builtins_static.a", "libnvptxcompiler_static.a"]);

/** Check the actual closure, including linker-option paths and short -l spellings, not only
 * the descriptor's explicit archive array. @param {string[]} inputs */
function rejectStaticNvrtcLinks(inputs) {
  const tokens = inputs.flatMap(input => input.startsWith("-Wl,") ? input.slice(4).split(",") : [input]);
  if (tokens.some(token => {
    const name = path.basename(token.replace(/^-l:?/, "")).toLowerCase();
    return staticNvrtcArchiveNames.has(name) || staticNvrtcArchiveNames.has(`lib${name}.a`);
  })) {throw new Error("Embedded CUDA images cannot be combined with static NVRTC archives");}
}

/** @typedef {{runner?: string[], env?: NodeJS.ProcessEnv}} Runner
 * @typedef {Runner & {args: string[]}} Profile
 * @typedef {{libraries: string[], links: string[], hashes: Record<string, string>,
 * receipts: Record<string, {path: string, sha256: string}>, noticeSources?: {component: string, path: string, sha256: string}[],
 * cudaImageInputs?: Record<string, string>}} NativeSdkSourceInput
 * Source-composed Linux descriptors name producer-admitted SDK roots, not raw archives.
 * @typedef {{acpp: string, hermes: string, rocm: string, nvrtc?: string, cudaImages?: string, zstd: string, openmpSdk: string}} SdkSources
 * @typedef {typeof import('./native-release-sdk-inputs') & {nativeZstdInputs: typeof import('./build-native-zstd').nativeZstdInputs,
 * nativeOpenSslInputs: typeof import('./build-native-openssl').nativeOpenSslInputs,
 * cudaImageInputs?: typeof import('./native-cuda-images').cudaImageInputs}} NativeSourceReaders
 * @typedef {{version: number, cpu: Runner & {cache: string, prepareCache?: boolean}, gpu: Record<string, Profile>,
 * components?: {core: string, gpu: Record<string,string>},
 * sdkSources?: SdkSources,
 * controller: Runner & {source: string, build: string, openssl: string, io: string, cc: string, nvml: string},
 * sdk: {defaultCompiler: string, cxx: string, flags: string[], libraries: string[], links: string[], sdkObjects: {object: string}[], dpcppComponent?: string}}} NativeSdk */

/** Build configuration is trusted compiler input, not pool input; still keep tokens literal
 * and diagnostics bounded. Runner prefixes activate an installed SDK/OS baseline without
 * shell evaluation or a second controller/kernel implementation.
 * @param {unknown} value @returns {value is string} */
function absolutePath(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}
/** Resolve a future output through symlinked existing ancestors without creating missing ones.
 * @param {string} file */
function prospectiveRealPath(file) {
  let existing = path.resolve(file);
  const suffix = [];
  while (!fs.lstatSync(existing, {throwIfNoEntry: false})) {
    const parent = path.dirname(existing);
    if (parent === existing) {break;}
    suffix.unshift(path.basename(existing));
    existing = parent;
  }
  return path.resolve(fs.realpathSync(existing), ...suffix);
}
/** Keep output out of maintained source roots, including aliases through symlinked parents.
 * @param {string} output @param {string} repo */
function assertNativeOutputOutsideSources(output, repo) {
  const canonicalRepo = fs.realpathSync(repo), canonicalOutput = prospectiveRealPath(output);
  for (const name of ["native", "sycl", "xmrig", "runtime", "scripts", "tests"]) {
    const sourcePath = path.join(canonicalRepo, name), stat = fs.lstatSync(sourcePath, {throwIfNoEntry: false});
    const canonicalSource = stat ? fs.realpathSync(sourcePath) : sourcePath;
    const relative = path.relative(canonicalSource, canonicalOutput);
    if (!relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {
      throw new Error("Native build output cannot enter a maintained source tree");
    }
  }
}
/** @param {Runner} runner */
function validateRunner(runner) {
  if (!runner || typeof runner !== "object" || (runner.runner !== undefined &&
      (!Array.isArray(runner.runner) || runner.runner.length > 32 || (runner.runner.length && !absolutePath(runner.runner[0])) ||
      runner.runner.some(value => typeof value !== "string" || value.length > 4096 || value.includes("\0"))))) {throw new Error("Invalid native SDK runner");}
  if (runner.env !== undefined && (!runner.env || typeof runner.env !== "object" || Array.isArray(runner.env) ||
      Object.entries(runner.env).some(([key, value]) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ||
        typeof value !== "string" || value.length > 8192 || value.includes("\0")))) {throw new Error("Invalid native SDK environment");}
  releaseBuildEnvironment(runner.env || {});
}
/** Validate the complete plan before creating output or starting compilers. Factory names
 * come from the maintained dispatcher; new compiler profiles require an explicit build recipe.
 * @param {unknown} value @param {string} repo */
function nativeSdkConfig(value, repo) {
  const sdk = /** @type {NativeSdk} */ (value);
  if (!sdk || sdk.version !== 1 || !sdk.sdk || ![sdk.sdk.flags, sdk.sdk.libraries, sdk.sdk.links].every(items =>
    Array.isArray(items) && items.length <= 4096 && items.every(item => typeof item === "string" && item.length <= 8192 && !/[\r\n\0]/.test(item))) ||
      !sdk.sdk.libraries.every(absolutePath) || !absolutePath(sdk.sdk.cxx) || !Array.isArray(sdk.sdk.sdkObjects) || sdk.sdk.sdkObjects.length > 64 ||
      sdk.sdk.sdkObjects.some(row => !row || !absolutePath(row.object)) ||
      Object.hasOwn(sdk.sdk, "objects") || Object.hasOwn(sdk.sdk, "receipts") ||
      sdk.sdk.dpcppComponent !== undefined && !absolutePath(sdk.sdk.dpcppComponent)) {throw new Error("Invalid native SDK link descriptor");}
  validateSdkSources(sdk.sdkSources);
  const {factories} = nativeApplicationGpu(fs.readFileSync(path.join(repo, "runtime/gpu-dispatch.cpp"), "utf8"), sdk.sdk);
  const keys = factories.map(name => name.replace("mom_native_gpu_", "").replaceAll("_", "-"));
  if (!sdk.gpu || typeof sdk.gpu !== "object" || Object.keys(sdk.gpu).length !== keys.length || keys.some(key => !Object.hasOwn(sdk.gpu, key))) {
    throw new Error("Native SDK must build every maintained GPU compiler profile");
  }
  for (const key of keys) {
    const profile = sdk.gpu[key];
    if (!profile) {throw new Error("Missing native SDK compiler profile");}
    validateRunner(profile);
    const count = key.startsWith("acpp-") || key === "dpcpp" ? 2 : ["oneapi", "dpcpp-opencl"].includes(key) ? 1 : 0;
    if (!count || !Array.isArray(profile?.args) || profile.args.length !== count || !profile.args.every(absolutePath)) {throw new Error("Invalid native SDK source roots: " + key);}
  }
  cudaImageReleaseMode(sdk, sdk.sdkSources);
  validateRunner(sdk.cpu); validateRunner(sdk.controller);
  if (!absolutePath(sdk.cpu.cache) || (sdk.cpu.prepareCache !== undefined && typeof sdk.cpu.prepareCache !== "boolean") ||
      ![sdk.controller.source, sdk.controller.build, sdk.controller.openssl, sdk.controller.io, sdk.controller.cc, sdk.controller.nvml].every(absolutePath)) {
    throw new Error("Invalid native Core/controller SDK roots");
  }
  if (sdk.cpu.prepareCache !== false && path.resolve(sdk.cpu.cache) !== path.join(repo, "build/cache/oneapi")) {
    throw new Error("Automatic preparation requires the shared Engineering cache; custom completed caches need prepareCache: false");
  }
  if (sdk.components !== undefined && (!sdk.components || Object.keys(sdk.components).sort().join(",") !== "core,gpu" ||
      !absolutePath(sdk.components.core) || !sdk.components.gpu || typeof sdk.components.gpu !== "object" || Array.isArray(sdk.components.gpu) ||
      Object.keys(sdk.components.gpu).sort().join(",") !== [...keys].sort().join(",") || !Object.values(sdk.components.gpu).every(absolutePath))) {
    throw new Error("Native component assembly requires an explicit Core and receipt for every GPU profile");
  }
  return {sdk, keys};
}

/** Source SDK receipts are distinct from GPU factory receipts. Keep this optional for
 * low-level build probes; the source-composed release route requires the complete cohort.
 * @param {SdkSources | undefined} sources */
function validateSdkSources(sources) {
  if (sources === undefined) {return;}
  const keys = Object.keys(sources || {}).sort().join(",");
  if (!sources || !["acpp,hermes,nvrtc,openmpSdk,rocm,zstd", "acpp,cudaImages,hermes,openmpSdk,rocm,zstd"].includes(keys) ||
      !Object.values(sources).every(absolutePath)) {throw new Error("Native SDK sources require explicit ACPP, Hermes, ROCm, OpenMP, either NVRTC or CUDA images, and zstd SDK roots");}
}

/** Native source releases may replace static NVRTC only when both CUDA graphs select one bundle.
 * @param {NativeSdk} sdk @param {SdkSources | undefined} sources @returns {boolean} */
function cudaImageReleaseMode(sdk, sources) {
  const selected = Object.entries(sdk.gpu || {}).filter(([, profile]) => profile?.env?.["MOM_AOT_CUDA_IMAGES"] !== undefined);
  if (!sources) {
    if (selected.length) {throw new Error("MOM_AOT_CUDA_IMAGES requires admitted sdkSources.cudaImages");}
    return false;
  }
  const bundle = sources["cudaImages"];
  const dpcpp = sdk.gpu?.["dpcpp"]?.env?.["MOM_AOT_CUDA_IMAGES"];
  const acpp = sdk.gpu?.["acpp-cuda"]?.env?.["MOM_AOT_CUDA_IMAGES"];
  if (bundle !== undefined) {
    if (!absolutePath(bundle) || dpcpp !== bundle || acpp !== bundle || selected.length !== 2 ||
        selected.some(([key]) => key !== "dpcpp" && key !== "acpp-cuda")) {
      throw new Error("CUDA image SDK sources require dpcpp and acpp-cuda MOM_AOT_CUDA_IMAGES to match the admitted bundle");
    }
    return true;
  }
  if (selected.length) {throw new Error("MOM_AOT_CUDA_IMAGES requires sdkSources.cudaImages and both CUDA profiles");}
  return false;
}

/** Derive the compiler closure from its producers, never a second LLVM library registry.
 * Descriptor libraries/links now contain only application-specific additions. Readers are
 * lazy because the original SDK producers reuse this module's path/build helpers.
 * @param {NativeSdk} sdk @param {NativeSourceReaders} [readers] @returns {NativeSdkSourceInput} */
function nativeSdkSourceInputs(sdk, readers) {
  validateSdkSources(sdk.sdkSources);
  const sources = sdk.sdkSources;
  if (!sources) {return {libraries: sdk.sdk.libraries, links: sdk.sdk.links, hashes: {}, receipts: {}};}
  const selectedReaders = readers || {...require("./native-release-sdk-inputs"), nativeZstdInputs: require("./build-native-zstd").nativeZstdInputs,
    nativeOpenSslInputs: require("./build-native-openssl").nativeOpenSslInputs};
  if (typeof selectedReaders.nativeZstdInputs !== "function" || typeof selectedReaders.nativeOpenSslInputs !== "function") {
    throw new Error("Source-composed native SDKs require zstd and OpenSSL receipt admission");
  }
  const imageMode = cudaImageReleaseMode(sdk, sources);
  if (imageMode) {rejectStaticNvrtcLinks([...sdk.sdk.libraries, ...sdk.sdk.links]);}
  const zstdRoot = fs.lstatSync(sources.zstd, {throwIfNoEntry: false});
  if (!zstdRoot?.isDirectory() || zstdRoot.isSymbolicLink()) {throw new Error("Native source zstd input must be a producer-admitted SDK root, not a raw archive");}
  const runtime = selectedReaders.nativeStaticRuntimeInputs(sources.acpp, {openmpSdkRoot: sources.openmpSdk});
  const openmp = runtime.openmp, openmpArchive = path.join(sources.openmpSdk, "sdk/libomp.a"),
    openmpReceipt = path.join(sources.openmpSdk, "result.json"), openmpBinding = runtime.pathBindings?.["/openmp/libomp.a"];
  if (!openmp || openmp.archive !== openmpArchive || openmp.receipt !== openmpReceipt ||
      openmp.files?.[openmpArchive] !== runtime.files?.[openmpArchive] ||
      openmp.files?.[openmpReceipt] !== openmp.receiptSha256 || !runtime.libraries.includes(openmpArchive) ||
      openmpBinding !== undefined && openmpBinding !== openmpArchive ||
      fileHash(openmpArchive) !== openmp.files[openmpArchive] ||
      fileHash(openmpReceipt) !== openmp.receiptSha256) {
    throw new Error("Native source SDK does not resolve ACPP's OpenMP input to the admitted SDK archive");
  }
  const openmpLicense = path.join(sources.openmpSdk, "source/openmp/LICENSE.TXT"),
    admittedOpenmp = /** @type {typeof openmp & {notice?: {path: string, sha256: string, bytes: number}}} */ (/** @type {unknown} */ (openmp)),
    openmpNotice = admittedOpenmp.notice;
  const openmpLicenseStat = fs.lstatSync(openmpLicense);
  if (!openmpNotice || Object.keys(openmpNotice).sort().join(",") !== "bytes,path,sha256" ||
      openmpNotice.path !== "source/openmp/LICENSE.TXT" || !Number.isSafeInteger(openmpNotice.bytes) || openmpNotice.bytes < 1 ||
      !/^[0-9a-f]{64}$/.test(openmpNotice.sha256 || "") || openmp.files?.[openmpLicense] !== openmpNotice.sha256 ||
      !openmpLicenseStat.isFile() || openmpLicenseStat.isSymbolicLink() || fs.realpathSync(openmpLicense) !== openmpLicense ||
      openmpLicenseStat.size !== openmpNotice.bytes || fileHash(openmpLicense) !== openmpNotice.sha256) {
    throw new Error("Native source SDK OpenMP notice differs from its admitted source body");
  }
  const openmpNotices = [{component: "LLVM OpenMP runtime", path: openmpLicense, sha256: openmpNotice.sha256}];
  const hermes = selectedReaders.nativeHermesInputs(sources.hermes), rocm = selectedReaders.nativeRocmHostInputs(sources.rocm),
    zstd = selectedReaders.nativeZstdInputs(sources.zstd);
  const zstdNotices = ["LICENSE", "COPYING"].map(name => {
    const file = path.join(sources.zstd, "source", name), sha256 = zstd.files?.[file];
    if (typeof sha256 !== "string" || !/^[0-9a-f]{64}$/.test(sha256)) {throw new Error("Native source SDK zstd notice is absent from admitted inputs: " + name);}
    return {component: "zstd " + name, path: file, sha256};
  });
  if (!Array.isArray(zstd.notices) || !isDeepStrictEqual(zstd.notices, zstdNotices)) {
    throw new Error("Native source SDK zstd notice rows differ from admitted source bodies");
  }
  const opensslRoot = sdk.controller.openssl, opensslReceipt = path.join(opensslRoot, "result.json"),
    openssl = selectedReaders.nativeOpenSslInputs(opensslRoot, "linux"), opensslLicense = path.join(opensslRoot, "source/LICENSE.txt"),
    opensslLibraries = [path.join(opensslRoot, "lib/libssl.a"), path.join(opensslRoot, "lib/libcrypto.a")];
  if (!openssl || typeof openssl !== "object" || Array.isArray(openssl) || openssl.receipt !== opensslReceipt ||
      !/^[0-9a-f]{64}$/.test(openssl.receiptSha256 || "") || !openssl.files || typeof openssl.files !== "object" ||
      Array.isArray(openssl.files) || openssl.files[opensslReceipt] !== openssl.receiptSha256 || Object.keys(openssl.files).length > 100000) {
    throw new Error("Native OpenSSL receipt admission is incomplete or selects another root");
  }
  if (!Array.isArray(openssl.libraries) || openssl.libraries.length !== opensslLibraries.length ||
      new Set(openssl.libraries).size !== openssl.libraries.length || opensslLibraries.some(file => !openssl.libraries.includes(file))) {
    throw new Error("Native OpenSSL archives differ from the maintained controller link inputs");
  }
  if (opensslLibraries.some(file => !/^[0-9a-f]{64}$/.test(openssl.files[file] || ""))) {
    throw new Error("Native OpenSSL admitted files omit a controller-selected static archive");
  }
  const opensslNoticeSha256 = openssl.files[opensslLicense];
  if (typeof opensslNoticeSha256 !== "string" || !/^[0-9a-f]{64}$/.test(opensslNoticeSha256)) {
    throw new Error("Native OpenSSL notice body is absent from admitted files");
  }
  const opensslNotices = [{component: "OpenSSL LICENSE", path: opensslLicense, sha256: opensslNoticeSha256}];
  if (!Array.isArray(openssl.notices) || !isDeepStrictEqual(openssl.notices, opensslNotices)) {
    throw new Error("Native OpenSSL notice rows differ from its admitted source body");
  }
  let nvrtc;
  if (!imageMode) {
    if (!sources.nvrtc) {throw new Error("Native source SDK omits its NVRTC receipt");}
    nvrtc = selectedReaders.nativeNvrtcInputs(sources.nvrtc, {requireVendorProvenance: true});
  }
  let cudaImages;
  if (imageMode) {
    const imageReader = selectedReaders.cudaImageInputs || require("./native-cuda-images").cudaImageInputs;
    const bundle = sources.cudaImages;
    if (!bundle) {throw new Error("Native source SDK omits its CUDA image bundle");}
    const admitted = imageReader(bundle, path.resolve(__dirname, ".."));
    const files = admitted?.files, sidecar = bundle + ".json";
    const bundleSha256 = files?.[bundle], sidecarSha256 = files?.[sidecar];
    if (!files || Object.keys(files).sort().join("\n") !== [bundle, sidecar].sort().join("\n") ||
        typeof bundleSha256 !== "string" || !/^[0-9a-f]{64}$/.test(bundleSha256) ||
        typeof sidecarSha256 !== "string" || !/^[0-9a-f]{64}$/.test(sidecarSha256) ||
        fileHash(bundle) !== bundleSha256 || fileHash(sidecar) !== sidecarSha256) {
      throw new Error("CUDA image SDK admission must bind the current include and sidecar files");
    }
    cudaImages = {receipt: sidecar, receiptSha256: sidecarSha256, files};
  }
  if (runtime.backend !== "cuda-hip" || hermes.source !== sdk.controller.source || hermes.build !== sdk.controller.build ||
      runtime.source !== sdk.gpu["acpp-cuda"]?.args[0] || runtime.source !== sdk.gpu["acpp-hip"]?.args[0] ||
      !runtime.comgr || runtime.comgr.receipt !== rocm.comgr.receipt ||
      rocm.libraries.some(file => !runtime.libraries.includes(file))) {
    throw new Error("Native source SDKs do not describe the selected controller and CUDA/HIP runtime cohort");
  }
  const hashes = /** @type {Record<string, string>} */ ({}), receipts = /** @type {Record<string, {path: string, sha256: string}>} */ ({});
  for (const [name, admitted] of Object.entries({acpp: runtime, openmp, hermes, rocm, ...(nvrtc ? {nvrtc} : {}),
    ...(cudaImages ? {cudaImages} : {}), zstd, openssl})) {
    receipts[name] = {path: admitted.receipt, sha256: admitted.receiptSha256};
    for (const [file, hash] of Object.entries({...admitted.files, [admitted.receipt]: admitted.receiptSha256})) {
      if (hashes[file] !== undefined && hashes[file] !== hash) {throw new Error("Conflicting native SDK receipt input: " + file);}
      hashes[file] = hash;
    }
  }
  // ACPP can discover generic NUMA as well as ROCm's exact staged copy. Only the
  // producer-proven identical original/staged pair may collapse; never dedupe by basename.
  const libraries = [...runtime.libraries.filter(file => file !== rocm.numa.original || file === rocm.numa.staged),
    ...(nvrtc?.archives || []), ...sdk.sdk.libraries];
  if (!libraries.includes(rocm.numa.staged) || new Set(libraries).size !== libraries.length) {
    throw new Error("Native source SDK archives are duplicated or omit ROCm's staged NUMA input");
  }
  const io = nativeIoInputs(sdk.controller.io), zlib = path.join(sdk.controller.io, "lib/libz.a");
  const zstdArchive = path.join(sources.zstd, "lib/libzstd.a"), zstdReceipt = path.join(sources.zstd, "result.json");
  if (!io.libraries.includes(zlib) || zstd.archive !== zstdArchive || zstd.receipt !== zstdReceipt ||
      !zstd.files || typeof zstd.files !== "object" || zstd.files[zstd.archive] !== fileHash(zstd.archive) ||
      zstd.files[zstd.receipt] !== zstd.receiptSha256 || !/^[0-9a-f]{64}$/.test(zstd.receiptSha256) ||
      zstd.artifacts?.["lib/libzstd.a"] !== zstd.files[zstd.archive] || !isStaticArchive(zstd.archive)) {
    throw new Error("Native source SDK requires an admitted static zstd SDK archive and explicit zlib");
  }
  const prefix = sdk.sdk.links.filter(flag => /^(?:--ld-path=|-Wl,--(?:threads|thinlto-jobs)=)/.test(flag));
  const suffix = sdk.sdk.links.filter(flag => !prefix.includes(flag));
  const links = [...prefix, ...runtime.links.map(flag => flag === "-lz" ? zlib : flag === "-lzstd" ? zstd.archive :
    runtime.resolvedLinks[flag] || flag), ...suffix];
  if (imageMode) {rejectStaticNvrtcLinks([...libraries, ...links]);}
  hashes[io.receipt] = io.receiptSha256;
  for (const [name, hash] of Object.entries(io.artifacts)) {hashes[path.join(sdk.controller.io, name)] = hash;}
  for (const [file, hash] of Object.entries(zstd.files)) {
    if (fileHash(file) !== hash) {throw new Error("Native zstd SDK input changed during assembly: " + file);}
  }
  const noticeSources = [...openmpNotices, ...zstdNotices, ...opensslNotices];
  for (const notice of noticeSources) {
    if (hashes[notice.path] !== notice.sha256 || fileHash(notice.path) !== notice.sha256) {
      throw new Error("Native source notice is absent from admitted link hashes: " + notice.component);
    }
  }
  // Preserve both admitted zlib and zstd closures; refreshing either could bless a post-admission swap.
  return {libraries, links, hashes, receipts, noticeSources,
    ...(cudaImages ? {cudaImageInputs: cudaImages.files} : {})};
}

/** Assemble separately produced CI components without recompiling them. Reuse the final
 * controller's source/portability/COMGR admission, not a second weaker cache policy. Hash
 * receipts and objects across assembly as well; no source receipt is generated or restamped.
 * @param {{core: string, gpu: Record<string,string>}} components @param {string[]} keys
 * @param {NativeSdk['sdk']} sdk @param {string} root @param {Record<string, string>} [cudaImageInputs]
 * @param {Parameters<typeof nativeApplicationInputs>[4]} [readImages] */
function nativeReleaseComponents(components, keys, sdk, root, cudaImageInputs, readImages) {
  const receipts = keys.map(key => components.gpu[key] || ""), objects = receipts.map((file, i) => {
    const record = readBuildJson(file);
    if (record?.factory !== "mom_native_gpu_" + keys[i]?.replaceAll("-", "_") || !absolutePath(record.object)) {throw new Error("Unexpected assembled GPU factory");}
    const key = keys[i] || "";
    if (cudaImageInputs && (key === "dpcpp" || key === "acpp-cuda")) {verifyCudaImageReceiptInputs(record, cudaImageInputs, key);}
    return {object: record.object};
  });
  const inputs = {...sdk, objects, receipts, ...(cudaImageInputs ? {cudaImageInputs} : {})}, factories = keys.map(key => "mom_native_gpu_" + key.replaceAll("-", "_"));
  const provenance = nativeApplicationInputs(root, components.core, inputs, factories, readImages);
  const hashes = buildFileInputs([components.core, path.join(path.dirname(components.core), "inputs.json"), ...receipts, ...objects.map(row => row.object)]);
  return {core: components.core, objects, receipts, hashes, provenance};
}

/** Resolve SDK short libraries to exact static archives rather than mutable -L search
 * results. OS/compiler support libraries keep the baseline compiler's normal lookup; their
 * provenance is a separate SDK gate. Hash only explicit files, never walk toolchain trees.
 * @param {NativeSdk} sdk @param {NativeSourceReaders} [readers] */
function nativeSdkLinkInputs(sdk, readers) {
  const sources = nativeSdkSourceInputs(sdk, readers);
  const component = sdk.sdk.dpcppComponent ? dpcppComponentInputs(sdk.sdk.dpcppComponent) : null;
  if (component) {
    const names = new Set(component.links.filter(absolutePath).map(file => path.basename(file)));
    if ([...sources.links, ...sources.libraries].some(file => absolutePath(file) && names.has(path.basename(file)))) {
      throw new Error("Do not mix a DPC++ component with manual copies of its runtime libraries");
    }
  }
  const sdkObjects = [...(component?.sdkObjects || []), ...sdk.sdk.sdkObjects];
  if (new Set(sdkObjects.map(row => row.object)).size !== sdkObjects.length) {throw new Error("Native SDK objects must not be duplicated");}
  const roots = /** @type {string[]} */ ([]), files = new Set([sdk.sdk.cxx, sdk.controller.cc, sdk.controller.nvml,
    ...sources.libraries, ...sdkObjects.map(row => row.object), ...Object.keys(component?.files || {}), ...Object.keys(sources.hashes)]);
  for (const runner of [sdk.cpu, sdk.controller, ...Object.values(sdk.gpu)]) {if (runner.runner?.[0]) {files.add(runner.runner[0]);}}
  for (let i = 0; i < sources.links.length; i++) {
    const flag = sources.links[i] || "";
    if (flag === "-L") {const root = sources.links[++i]; if (!absolutePath(root)) {throw new Error("Invalid SDK library search root");} roots.push(root);} else if (flag.startsWith("-L")) {
      const root = flag.slice(2); if (!absolutePath(root)) {throw new Error("Invalid SDK library search root");} roots.push(root);
    }
  }
  const support = new Set(["dl", "pthread", "m", "z", "zstd", "stdc++fs"]);
  for (const file of sources.libraries) {if (!isStaticArchive(file)) {throw new Error("Native SDK library is not a regular static archive");}}
  const allLinks = [...(component?.links || []), ...sources.links];
  const links = allLinks.map((flag, i) => {
    if (flag.startsWith("-l") && !support.has(flag.slice(2))) {
      const name = flag.slice(2);
      if (!/^[A-Za-z0-9_+.-]{1,128}$/.test(name)) {throw new Error("Invalid native SDK library name");}
      const library = roots.map(root => path.join(root, "lib" + name + ".a")).find(file => fs.existsSync(file));
      if (!library) {throw new Error("Missing static native SDK library: " + name);}
      if (!isStaticArchive(library)) {throw new Error("Resolved native SDK library is not a regular static archive");}
      files.add(library); return library;
    }
    if (absolutePath(flag) && allLinks[i - 1] !== "-L") {
      if (flag.endsWith(".a") && !isStaticArchive(flag)) {throw new Error("Native SDK link input is not a regular static archive");}
      files.add(flag);
    }
    if (flag.startsWith("-Wl,")) {for (const token of flag.slice(4).split(",")) {if (absolutePath(token)) {files.add(token);}}}
    if (flag.startsWith("--ld-path=")) {files.add(flag.slice(10));}
    return flag;
  });
  if (sources.cudaImageInputs) {rejectStaticNvrtcLinks([...sources.libraries, ...links]);}
  const hashes = Object.fromEntries([...files].map(file => [file, fileHash(file)]));
  for (const [file, hash] of Object.entries(sources.hashes)) {
    if (hashes[file] !== hash) {throw new Error("Native source SDK input changed during assembly: " + file);}
  }
  for (const notice of sources.noticeSources || []) {
    if (hashes[notice.path] !== notice.sha256) {throw new Error("Native source notice differs from final link hashes: " + notice.component);}
  }
  return {links, libraries: sources.libraries, sdkObjects, hashes, ...(sdk.sdkSources ? {sources: sources.receipts,
    noticeSources: sources.noticeSources} : {}),
  ...(sources.cudaImageInputs ? {cudaImageInputs: sources.cudaImageInputs} : {})};
}

/** Read only the final stage-log bytes for actionable failure output; diagnostics must not
 * replace the primary compiler failure if the log itself is unreadable.
 * @param {number} fd @returns {string} */
function nativeStageLogTail(fd) {
  const limit = 8 * 1024;
  try {
    const size = fs.fstatSync(fd).size;
    if (!Number.isSafeInteger(size) || size < 0) {return "(stage log tail unavailable)";}
    const length = Math.min(size, limit), buffer = Buffer.alloc(length);
    let offset = 0;
    while (offset < length) {
      const count = fs.readSync(fd, buffer, offset, length - offset, size - length + offset);
      if (!count) {break;}
      offset += count;
    }
    return (size > limit ? "[earlier stage output omitted]\n" : "") + (offset ? buffer.subarray(0, offset).toString("utf8") : "(stage log is empty)");
  } catch {return "(unable to read stage log tail)";}
}

/** A stage invokes only the maintained producer. Output is streamed to disk, not accumulated
 * in a parent Node buffer. Never retry a failed/uncertain compiler lifecycle automatically.
 * @param {Runner} runner @param {string} command @param {string[]} args @param {NodeJS.ProcessEnv} env
 * @param {string} repo @param {string} output @param {string} stage @param {typeof spawnSync} spawn */
function nativeStage(runner, command, args, env, repo, output, stage, spawn = spawnSync) {
  const prefix = runner.runner || [], executable = prefix[0] || command, tokens = prefix.length ? [...prefix.slice(1), command, ...args] : args;
  const selected = {...env, ...runner.env}, buildEnv = releaseBuildEnvironment(selected);
  // All graphs are explicit in the orchestration plan. A cache-preparation stage may select
  // only one compiler without relaxing portability, cache admission or worker-reuse policy.
  for (const key of ["MOM_GPU_BACKEND", "MOM_LINUX_BUILD_COMPILER"]) {if (selected[key]) {buildEnv[key] = selected[key];}}
  const log = path.join(output, "logs", stage + ".log"), fd = fs.openSync(log, "wx+");
  console.log("[native] " + stage);
  try {
    const result = spawn(executable, tokens, {cwd: repo, stdio: ["ignore", fd, fd],
      env: {...buildEnv, MOM_AOT_BUILD_ONLY: "1", MOM_NATIVE_BUILD_OUTPUT: output, MOM_NATIVE_BUILD_STAGE: stage, GIT_OPTIONAL_LOCKS: "0"}});
    if (result.error || result.signal || result.status !== 0) {
      const error = result.error?.message ? result.error.message.slice(0, 1024) + (result.error.message.length > 1024 ? "…" : "") : "none";
      console.error(`[native] Failed stage ${stage} (exit ${result.status}, signal ${result.signal || "none"}, error ${error}); output from ${log}:\n${nativeStageLogTail(fd)}`);
      throw new Error("Native build failed: " + stage + "; inspect " + log + " before retrying", {cause: result.error});
    }
  } finally {fs.closeSync(fd);}
}

/** Strip into an unpublished file and verify every runtime/image payload before promotion.
 * The receipt records the exact strip/readelf programs because their version shapes the bytes.
 * @param {string} original @param {string} output */
function compactNativeRelease(original, output) {
  const pending = path.join(output, "mom.pending"), binary = path.join(output, "mom");
  if ([pending, binary].some(file => fs.existsSync(file))) {throw new Error("Native compaction requires fresh output paths");}
  const tools = {strip: binutilsIdentity(stripTool), readelf: binutilsIdentity(readelfTool)}, before = nativeElf(original);
  // Driver defaults differ: a successful Intel link must not silently drop executable ASLR.
  // DYN alone also describes shared libraries, so require the executable's OS interpreter.
  if (!/^Elf file type is DYN\b/m.test(before.programHeaders) || !/^\s*INTERP\s/m.test(before.programHeaders)) {
    throw new Error("Native release must be a position-independent executable");
  }
  const stripped = spawnSync(stripTool, ["--strip-unneeded", "-o", pending, "--", original], {encoding: "utf8", timeout: 120000, maxBuffer: 1 << 20});
  if (stripped.error || stripped.signal || stripped.status !== 0) {throw new Error("Native executable stripping failed", {cause: stripped.error});}
  const after = nativeElf(pending); verifyNativeStrip(before, after);
  if (Object.values(tools).some(tool => fileHash(tool.path) !== tool.sha256)) {throw new Error("Native binutils changed during compaction");}
  const needed = nativeDependencies(elfReport(pending, "--dynamic")), glibc = auditNativeGlibc(elfReport(pending, "--version-info"), "2.39");
  return {binary, pendingBinary: pending, sha256: after.sha256, bytes: after.bytes, needed, glibc,
    originalSha256: before.sha256, sections: after.sections, tools};
}

/** Compile one maintained source codebase using installed pinned SDKs. A new explicit output
 * root protects existing artifacts; SDK installation and full release qualification remain
 * separate gates. No VM, GPU execution, publication or Engineering-path replacement occurs.
 * @param {string} configFile @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildNativeRelease(configFile, output, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Native application orchestration currently supports Linux x86-64 only; Windows is not qualified");}
  if (!absolutePath(configFile) || !absolutePath(output) || fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("Select an absolute SDK descriptor and a new absolute native output directory");}
  const env = /** @type {NodeJS.ProcessEnv} */ ({...releaseBuildEnvironment(environment), MOM_AOT_BUILD_ONLY: "1"});
  const repo = path.resolve(__dirname, "..");
  assertNativeOutputOutsideSources(output, repo);
  const configHash = fileHash(configFile), {sdk, keys} = nativeSdkConfig(readBuildJson(configFile), repo);
  if (fileHash(configFile) !== configHash) {throw new Error("Native SDK descriptor changed during admission");}
  // Image selection is profile-local and receipt-bound; ambient shell state must not turn an
  // unrelated profile into an image consumer or bypass the descriptor's source admission.
  delete env["MOM_AOT_CUDA_IMAGES"];
  const recipe = sourceInputs(repo, [], ["scripts/build-native-release.js", "scripts/build-native-application.js", "scripts/native-elf.js", "scripts/build-release.js", "scripts/build-environment.js", "scripts/build-native-io.js", "scripts/native-dpcpp-component.js", "scripts/native-release-sdk-inputs.js", "scripts/native-cuda-images.js"]);
  nativeIoInputs(sdk.controller.io); // Fail before costly Core/GPU compilation, not at the final controller link.
  const sdkInputs = nativeSdkLinkInputs(sdk); sdk.sdk.links = sdkInputs.links; sdk.sdk.libraries = sdkInputs.libraries; sdk.sdk.sdkObjects = sdkInputs.sdkObjects;
  const components = sdk.components ? nativeReleaseComponents(sdk.components, keys, sdk.sdk, repo, sdkInputs.cudaImageInputs) : null;
  if (fs.lstatSync(output, {throwIfNoEntry: false})) {throw new Error("Select an absolute SDK descriptor and a new absolute native output directory");}
  assertNativeOutputOutsideSources(output, repo);
  fs.mkdirSync(path.dirname(output), {recursive: true}); fs.mkdirSync(output);
  fs.mkdirSync(path.join(output, "logs")); fs.mkdirSync(path.join(output, "gpu")); fs.mkdirSync(path.join(output, "application"));
  if (!components && sdk.cpu.prepareCache !== false) {
    nativeStage({...sdk.cpu, env: {...sdk.cpu.env, MOM_LINUX_BUILD_COMPILER: "oneapi", MOM_GPU_BACKEND: "intel", MOM_PORTABLE_BUILD: "1"}},
      "bash", [path.join(repo, "scripts/multicompiler-entrypoint.sh"), "true"], env, repo, output, "engineering-cache");
  }
  const node = (/** @type {Runner} */ runner, /** @type {string} */ script, /** @type {string[]} */ args,
    /** @type {string} */ stage) => nativeStage(runner, process.execPath, ["--max-old-space-size=256", path.join(repo, "scripts", script), ...args], env, repo, output, stage);
  if (!components) {node(sdk.cpu, "build-native-core.js", [sdk.cpu.cache, path.join(output, "core"), "--gpu"], "core");}
  const core = components?.core || path.join(output, "core/cpu-core.a");
  const receipts = components?.receipts || [], objects = components?.objects || [];
  for (const key of (components ? [] : keys).sort((a, b) => Number(b === "oneapi") - Number(a === "oneapi") || Number(a.endsWith("hip")) - Number(b.endsWith("hip")))) {
    const profile = sdk.gpu[key], acpp = key.startsWith("acpp-"), directory = path.join(output, "gpu", key);
    node({...profile, env: {...profile?.env, ...(acpp ? {MOM_AOT_SYCL_BACKEND: key.slice(5)} : {MOM_AOT_DPCPP_PROFILE: key})}},
      acpp ? "native-gpu-build.js" : "build-native-gpu.js", [...(profile?.args || []), directory], key);
    const receipt = path.join(directory, "result.json"), record = readBuildJson(receipt);
    if (record.factory !== "mom_native_gpu_" + key.replaceAll("-", "_") || !absolutePath(record.object)) {throw new Error("Unexpected built native GPU factory: " + key);}
    if (sdkInputs.cudaImageInputs && (key === "dpcpp" || key === "acpp-cuda")) {
      verifyCudaImageReceiptInputs(record, sdkInputs.cudaImageInputs, key);
    }
    receipts.push(receipt); objects.push({object: record.object});
  }
  // Compiler basename selects its driver mode (clang++ versus clang). Invoke the admitted
  // SDK paths unchanged; renaming clang++ through a generic alias drops C++ runtime linkage.
  const manifest = path.join(output, "gpu-inputs.json");
  fs.writeFileSync(manifest, JSON.stringify({version: 1, ...sdk.sdk, objects, receipts,
    ...(sdkInputs.cudaImageInputs ? {cudaImageInputs: sdkInputs.cudaImageInputs} : {})}, null, 2) + "\n", {flag: "wx"});
  node({...sdk.controller, env: {...sdk.controller.env, CC: sdk.controller.cc, MOM_AOT_CPU_CORE: core,
    MOM_AOT_GPU_INPUTS: manifest, MOM_AOT_OPENSSL_ROOT: sdk.controller.openssl, MOM_AOT_IO_SDK: sdk.controller.io, MOM_AOT_NVML_HEADER: sdk.controller.nvml}},
  "build-native-application.js", [sdk.controller.source, sdk.controller.build, path.join(output, "application"), "--app"], "application");
  const builds = fs.readdirSync(path.join(output, "application")).filter(name => /^mom-aot-[A-Za-z0-9]{6}$/.test(name));
  if (builds.length !== 1) {throw new Error("Unrecognized native application build output");}
  const build = path.join(output, "application", builds[0] || ""), result = readBuildJson(path.join(build, "result.json"));
  if (result.application !== true || result.entrypoint !== "mom.js" || result.buildOnly !== true) {throw new Error("Native build did not compile the maintained application");}
  const {pendingBinary, ...artifact} = compactNativeRelease(path.join(build, "controller-native"), output);
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Native release recipe");
  if (Object.entries(sdkInputs.hashes).some(([file, sha256]) => fileHash(file) !== sha256)) {throw new Error("Native SDK link inputs changed during compilation");}
  if (Object.entries(components?.hashes || {}).some(([file, hash]) => fileHash(file) !== hash)) {throw new Error("Native component inputs changed during assembly");}
  if (fileHash(configFile) !== configHash) {throw new Error("Native SDK descriptor changed during compilation");}
  fs.renameSync(pendingBinary, artifact.binary);
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({version: require("../package.json").version, configHash, recipe, sdkInputs, ...artifact,
    application: result, ...(components ? {components} : {}), buildOnly: true, publicationQualified: false,
    scope: "Automatically compiled maintained JS/Core/five SYCL graphs; SDK sources/closure and complete Linux/GPU/Windows release qualification remain separate gates. No executable was run."}, null, 2) + "\n", {flag: "wx"});
  console.log("[native] Built " + artifact.binary + "; NOT executed or publication-qualified.");
  return artifact;
}

// Publish helpers before invoking the CLI: lazy SDK readers load original producer helpers
// which in turn import absolutePath from this module during a source-composed release build.
module.exports = {absolutePath, nativeSdkConfig, nativeSdkLinkInputs, nativeSdkSourceInputs, nativeReleaseComponents, nativeStage, compactNativeRelease, buildNativeRelease};

if (require.main === module) {
  try {
    const [config, output, extra] = process.argv.slice(2);
    if (!config || !output || extra) {throw new Error("Usage: node scripts/build-native-release.js /absolute/sdk.json /new/absolute/output");}
    buildNativeRelease(config, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Native build failed"); process.exitCode = 1;}
}
