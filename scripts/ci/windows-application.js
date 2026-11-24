#!/usr/bin/env node
"use strict";

// Final Windows application CI job: admit the external same-run pins, restore every pinned
// handoff, compose the current SDK descriptor, link the build-only mom.exe, and seal the raw
// executable plus compact build evidence. Workflow provisioning and artifact transfer stay in YAML.
const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {windowsHostSdkEnvironment, windowsHostSdkPaths} = require("../native-ci-windows-llvm");
const {hasExactBuildFields, fileHash, readCanonicalBuildJson} = require("../build-source-inputs");
const {applicationBuildFiles, applicationBuildName, ciSuffix, ciBuildJobs} = require("../native-ci-job");

const repo = path.resolve(__dirname, "../.."), intelRoot = "/opt/intel/oneapi/compiler/2026.2",
  profiles = ["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"],
  incomingGpuPinKeys = ["bundleSha256", "configSha256", "handoffSha256", "profile", "receiptSha256"],
  sealedGpuPinKeys = ["configSha256", "handoffSha256", "profile", "receiptSha256"],
  sha256Pattern = /^[a-f0-9]{64}$/,
  restoreReserve = 4n * 1024n ** 3n, GiB = 1024 ** 3, MiB = 1024 ** 2;

// Environment segment of each MOM_NATIVE_WINDOWS_APPLICATION_<NAME>_SHA256 pin.
const pinVariables = {
  coreBundle: "CORE_BUNDLE", coreHandoff: "CORE_HANDOFF", dpcppComponentBundle: "DPCPP_COMPONENT_BUNDLE",
  dpcppComponentManifest: "DPCPP_COMPONENT_MANIFEST", hostComponentBundle: "HOST_COMPONENT_BUNDLE",
  hostComponentManifest: "HOST_COMPONENT_MANIFEST", ioBundle: "IO_BUNDLE", ioHandoff: "IO_HANDOFF",
  hermesBundle: "HERMES_BUNDLE", hermesHandoff: "HERMES_HANDOFF", linuxHermesBundle: "LINUX_HERMES_BUNDLE",
  linuxToolsBundle: "LINUX_TOOLS_BUNDLE", linuxToolsHandoff: "LINUX_TOOLS_HANDOFF", dpcppInputsBundle: "DPCPP_INPUTS_BUNDLE",
  dpcppInputsManifest: "DPCPP_INPUTS_MANIFEST", linuxSourcesBundle: "LINUX_SOURCES_BUNDLE",
  hipHeadersBundle: "HIP_HEADERS_BUNDLE", hipHeadersHandoff: "HIP_HEADERS_HANDOFF", cudaImagesBundle: "CUDA_IMAGES_BUNDLE",
};

/** @typedef {Record<keyof typeof pinVariables, string>} ApplicationPins */
/** @typedef {{profile:string,bundleSha256:string,configSha256:string,handoffSha256:string,receiptSha256:string}} GpuPin */
/** @typedef {{key:string,file:string,pin:string,roots:string[],maxArchive:number,maxExpanded:number,destination:string}} InputRow */
/** @typedef {{dev:number,ino:number,size:number,mtimeMs:number}} FileIdentity */
/** @typedef {{row:InputRow,identity:FileIdentity,members:string[],expanded:number}} PinnedInput */
/** @typedef {ReturnType<typeof realBuildDependencies>} BuildDependencies */

/** Production readers and composers; tests substitute recorded fakes for the whole set. */
function realBuildDependencies() {
  return {
    ci: {...require("../native-ci-tar"), ...require("../native-ci-linux-sources"), ...require("../native-dpcpp-cuda-inputs"),
      ...require("../native-ci-linux-cuda"), ...require("../native-ci-linux-release"), ...require("../native-ci-windows-llvm")},
    sourceInputs: require("../build-source-inputs"),
    linuxLlvmToolHandoffInputs: require("../native-linux-llvm-tools").linuxLlvmToolHandoffInputs,
    windowsControllerHandoffInputs: require("../native-windows-controller-handoff").windowsControllerHandoffInputs,
    windowsNativeCoreHandoffInputs: require("../native-windows-core-handoff").windowsNativeCoreHandoffInputs,
    windowsDpcppComponentInputs: require("../native-windows-dpcpp-component").windowsDpcppComponentInputs,
    windowsHostComponentInputs: require("../native-windows-host-component").windowsHostComponentInputs,
    windowsGpuHandoffInputs: require("../native-windows-gpu-handoff").windowsGpuHandoffInputs,
    rocmHipHeaderInputs: require("../native-rocm-hip-headers").rocmHipHeaderInputs,
    nativeHermesInputs: require("../native-release-sdk-inputs").nativeHermesInputs,
    composeWindowsApplicationConfig: require("../native-windows-application-config").composeWindowsApplicationConfig,
    windowsSystemImports: require("../build-native-windows-controller").windowsSystemImports,
    windowsGpu: require("../run-native-windows-gpu-ci"),
    spawnSync: require("node:child_process").spawnSync,
  };
}

/** @param {unknown} value @returns {value is GpuPin} */
function incomingGpuPin(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {return false;}
  const pin = /** @type {Record<string, unknown>} */ (value);
  return hasExactBuildFields(pin, incomingGpuPinKeys) && typeof pin["profile"] === "string" &&
    profiles.includes(pin["profile"]) && [pin["bundleSha256"], pin["configSha256"], pin["handoffSha256"], pin["receiptSha256"]]
    .every(digest => typeof digest === "string" && sha256Pattern.test(digest));
}

/** Admit the 19 external bundle/receipt pins and exactly one GPU handoff pin per maintained profile.
 * @param {NodeJS.ProcessEnv} environment @returns {{pins:ApplicationPins,gpuPins:GpuPin[]}} */
function admitPins(environment) {
  const pins = /** @type {ApplicationPins} */ (Object.fromEntries(Object.entries(pinVariables).map(([key, name]) =>
    [key, environment[`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`] || ""])));
  if (Object.values(pins).some(pin => !sha256Pattern.test(pin))) {
    throw new Error("Missing or malformed external Windows application bundle/receipt pin");
  }
  const raw = environment["MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON"] || "";
  if (!raw || Buffer.byteLength(raw, "utf8") > 64 * 1024) {throw new Error("GPU handoff pins are empty or excessive");}
  /** @type {unknown} */
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {throw new Error("GPU handoff pins are invalid JSON", {cause: error});}
  if (!Array.isArray(parsed) || parsed.length !== profiles.length || !parsed.every(incomingGpuPin) ||
      new Set(parsed.map(pin => pin.profile)).size !== profiles.length) {
    throw new Error("GPU handoff pins must cover the exact five maintained profiles once each");
  }
  return {pins, gpuPins: parsed};
}

/** The workflow exports job start + timeout - 300 s as the one aggregate deadline (Unix seconds).
 * @param {NodeJS.ProcessEnv} environment @param {number} [nowMs] */
function ciDeadline(environment, nowMs = Date.now()) {
  const value = environment["MOM_CI_DEADLINE_EPOCH"] || "";
  if (!/^[0-9]{10}$/.test(value) || Number(value) * 1000 <= nowMs) {
    throw new Error("MOM_CI_DEADLINE_EPOCH aggregate deadline is invalid or expired");
  }
  return Number(value);
}

/** Bound one child operation by its cap and the aggregate deadline minus a reserve.
 * @param {number} deadline @param {number} capMs @param {number} reserveMs */
function remainingMs(deadline, capMs, reserveMs) {
  const milliseconds = Math.min(capMs, deadline * 1000 - Date.now() - reserveMs);
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 1000) {throw new Error("Windows application aggregate deadline expired");}
  return milliseconds;
}

/** Workflow command files must be one ordinary file, never a link to somewhere else. @param {string} file */
function assertWorkflowFile(file) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {throw new Error("Workflow output is not a canonical regular file");}
}

/** `pins` command: fail fast on malformed external pins before provisioning starts.
 * @param {NodeJS.ProcessEnv} [environment] */
function runPins(environment = process.env) {
  ciSuffix(environment);
  admitPins(environment);
}

/** Every flat download in RUNNER_TEMP/incoming with its pin, restore roots and size bounds.
 * @param {string} incoming @param {string} temp @param {string} suffix @param {ApplicationPins} pins @param {GpuPin[]} gpuPins
 * @param {{hermes:{sourceName:string,sdkName:string},dpcpp:{name:string,sdkTar:string},linuxSource:{sourceName:string,sourceTar:string}}} state
 * @returns {InputRow[]} */
function inputRows(incoming, temp, suffix, pins, gpuPins, state) {
  const toolName = `native-linux-llvm-tools-${suffix}`,
    /** @type {(key:string,file:string,pin:string,roots:string[],maxArchive:number,maxExpanded:number,destination?:string) => InputRow} */
    row = (key, file, pin, roots, maxArchive, maxExpanded, destination = "/tmp") =>
      ({key, file: path.join(incoming, file), pin, roots, maxArchive, maxExpanded, destination});
  return [
    row("core", `native-windows-core-${suffix}.tar`, pins.coreBundle, [`native-windows-core-${suffix}`], 512 * MiB, 512 * MiB),
    row("dpcppComponent", `native-windows-dpcpp-static-${suffix}.tar`, pins.dpcppComponentBundle,
      [`native-windows-dpcpp-static-${suffix}`], 2 * GiB, 2 * GiB),
    row("hostComponent", `native-windows-host-component-${suffix}.tar`, pins.hostComponentBundle,
      [`native-windows-host-component-${suffix}`], 2 * GiB, 2 * GiB),
    row("io", `native-windows-controller-io-${suffix}.tar`, pins.ioBundle, [`native-windows-controller-io-${suffix}`], GiB, GiB, temp),
    row("windowsHermes", `native-windows-controller-hermes-${suffix}.tar`, pins.hermesBundle,
      [`native-windows-controller-hermes-${suffix}`], GiB, GiB, temp),
    row("linuxHermes", `native-linux-hermes-sdk-${suffix}.tar`, pins.linuxHermesBundle,
      [state.hermes.sourceName, state.hermes.sdkName], 16 * GiB, 12 * GiB),
    row("linuxTools", `${toolName}.tar`, pins.linuxToolsBundle, [toolName], 2 * GiB, 2 * GiB, temp),
    row("dpcppInputs", path.basename(state.dpcpp.sdkTar), pins.dpcppInputsBundle, [state.dpcpp.name], 16 * GiB, 8 * GiB),
    row("linuxSources", path.basename(state.linuxSource.sourceTar), pins.linuxSourcesBundle,
      [state.linuxSource.sourceName], 16 * GiB, 8 * GiB),
    row("hipHeaders", `native-rocm-hip-headers-${suffix}.tar`, pins.hipHeadersBundle, [`native-rocm-hip-headers-${suffix}`], GiB, 128 * MiB),
    row("cudaImages", `native-linux-cuda-images-${suffix}.tar`, pins.cudaImagesBundle,
      [`mom-native-cuda-images-handoff-${suffix}`], GiB, GiB),
    ...profiles.map(profile => row(`gpu:${profile}`, `native-windows-gpu-${profile}-${suffix}.tar`,
      gpuPins.find(pin => pin.profile === profile)?.bundleSha256 || "", [`mom-native-windows-gpu-${profile}-${suffix}`], GiB, GiB)),
  ];
}

/** @param {fs.Stats} stat @returns {FileIdentity} */
function fileIdentity(stat) {
  return {dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs};
}

/** @param {fs.Stats} stat @param {FileIdentity} identity */
function sameIdentity(stat, identity) {
  return stat.dev === identity.dev && stat.ino === identity.ino && stat.size === identity.size && stat.mtimeMs === identity.mtimeMs;
}

/** Check every downloaded bundle pin before GNU tar parses even the first header, then bound
 * each archive's member set and expanded size. @param {InputRow[]} rows @param {number} deadline
 * @param {BuildDependencies} deps @returns {PinnedInput[]} */
function pinInputs(rows, deadline, deps) {
  const {ci, sourceInputs: {fileHash}} = deps;
  const pinned = rows.map(row => {
    const stat = fs.lstatSync(row.file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > row.maxArchive ||
        fs.realpathSync(row.file) !== row.file || fileHash(row.file) !== row.pin) {
      throw new Error("Windows application archive differs from its external same-run SHA-256: " + path.basename(row.file));
    }
    return {row, identity: fileIdentity(stat)};
  });
  return pinned.map(({row, identity}) => {
    const members = ci.listState(row.file, row.roots, remainingMs(deadline, 5 * 60 * 1000, 60000)),
      expanded = ci.tarExpandedBytes(row.file, members.length, remainingMs(deadline, 8 * 60 * 1000, 60000));
    if (expanded > row.maxExpanded) {throw new Error("Windows application input exceeds its extraction-size bound: " + row.key);}
    return {row, identity, members, expanded};
  });
}

/** All restores sharing one filesystem must leave the four-GiB build reserve free. @param {PinnedInput[]} inputs */
function assertAggregateSpace(inputs) {
  /** @type {Map<number, {destination:string,bytes:number}>} */
  const byDevice = new Map();
  for (const {row, expanded} of inputs) {
    const device = fs.statSync(row.destination).dev, prior = byDevice.get(device);
    byDevice.set(device, {destination: prior ? prior.destination : row.destination, bytes: (prior ? prior.bytes : 0) + expanded});
  }
  for (const {destination, bytes} of byDevice.values()) {
    const statfs = fs.statfsSync(destination, {bigint: true});
    if (BigInt(bytes) + restoreReserve > statfs.bavail * statfs.bsize) {
      throw new Error("Windows application aggregate restore would consume the four-GiB build reserve");
    }
  }
}

/** Extract one pinned input to its fresh roots and prove the transport did not change meanwhile.
 * @param {PinnedInput} input @param {number} deadline @param {BuildDependencies} deps */
function restoreInput(input, deadline, deps) {
  const {row, identity, expanded} = input, statfs = fs.statfsSync(row.destination, {bigint: true});
  if (BigInt(expanded) + restoreReserve > statfs.bavail * statfs.bsize) {
    throw new Error("Windows application input restore would consume the four-GiB build reserve: " + row.key);
  }
  for (const root of row.roots) {if (fs.existsSync(path.join(row.destination, root))) {throw new Error("Restore root already exists: " + root);}}
  deps.ci.extractState(row.file, row.destination, row.roots, row.roots, remainingMs(deadline, 15 * 60 * 1000, 60000));
  if (!sameIdentity(fs.lstatSync(row.file), identity) || deps.sourceInputs.fileHash(row.file) !== row.pin) {
    throw new Error("Windows application archive changed during extraction: " + path.basename(row.file));
  }
}

/** Admissions use restored payloads, not transport tars. Release only these exact verified
 * downloads before SDK acquisition/linking needs additional disk space.
 * @param {PinnedInput[]} inputs @param {BuildDependencies} deps */
function releaseTransports(inputs, deps) {
  for (const {row, identity} of inputs) {
    const stat = fs.lstatSync(row.file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || !sameIdentity(stat, identity) ||
        fs.realpathSync(row.file) !== row.file || deps.sourceInputs.fileHash(row.file) !== row.pin) {
      throw new Error("Windows application transport changed before cleanup: " + path.basename(row.file));
    }
    fs.unlinkSync(row.file);
  }
}

/** Re-admit each restored GPU factory handoff against its external receipt/config/handoff pins.
 * @param {string} suffix @param {GpuPin[]} gpuPins @param {BuildDependencies} deps */
function admitGpuHandoffs(suffix, gpuPins, deps) {
  const byProfile = new Map(gpuPins.map(pin => [pin.profile, pin])), gpuReceipts = [], gpuHandoffs = [];
  for (const profile of profiles) {
    const receipt = path.join("/tmp", `mom-native-windows-gpu-${profile}-${suffix}`, "output/result.json"), pin = byProfile.get(profile);
    if (!pin) {throw new Error("Missing Windows GPU profile bundle pin: " + profile);}
    const expected = {profile, receiptSha256: pin.receiptSha256, configSha256: pin.configSha256,
        handoffSha256: pin.handoffSha256}, admitted = deps.windowsGpuHandoffInputs(receipt, expected);
    gpuReceipts.push(receipt);
    gpuHandoffs.push(expected);
    if (admitted.profile !== profile) {throw new Error("GPU handoff reader returned another profile");}
  }
  return {gpuReceipts, gpuHandoffs};
}

/** Re-admit every restored handoff, reacquire the current Windows SDK, and write the composed
 * application descriptor exactly once. Returns the HIP header admission so the build can be
 * checked against it afterwards.
 * @param {{suffix:string,temp:string,configFile:string,pins:ApplicationPins,gpuPins:GpuPin[],
 * paths:ReturnType<BuildDependencies["ci"]["ciPaths"]>,environment:NodeJS.ProcessEnv}} plan @param {BuildDependencies} deps */
function composeApplication(plan, deps) {
  const {ci, sourceInputs: {fileHash}} = deps, {suffix, temp, pins, paths, environment} = plan,
    linuxSource = ci.linuxSourceStatePaths(paths, environment), dpcpp = ci.dpcppCudaInputsStatePaths(paths, environment),
    hermes = ci.hermesStatePaths(paths, environment);
  ci.verifyLinuxSources(paths, linuxSource, fs.realpathSync("/usr/bin/git"));
  const dpcppManifest = deps.windowsGpu.windowsGpuDpcppManifest(dpcpp.manifest, pins.dpcppInputsManifest, pins.linuxSourcesBundle),
    dpcppInputs = ci.auditDpcppCudaInputs(dpcpp, dpcppManifest, pins.dpcppInputsManifest),
    {gpuReceipts, gpuHandoffs} = admitGpuHandoffs(suffix, plan.gpuPins, deps);
  const coreRoot = path.join("/tmp", `native-windows-core-${suffix}`), hipRoot = path.join("/tmp", `native-rocm-hip-headers-${suffix}`),
    hipExpectation = {handoffSha256: pins.hipHeadersHandoff},
    hipHeaders = deps.rocmHipHeaderInputs(hipRoot, hipExpectation),
    core = deps.windowsNativeCoreHandoffInputs(coreRoot, {handoffSha256: pins.coreHandoff}),
    hermesRoot = path.join(temp, `native-windows-controller-hermes-${suffix}`),
    hermesExpectation = /** @type {const} */ ({kind: "hermes", handoffSha256: pins.hermesHandoff}),
    io = deps.windowsControllerHandoffInputs(path.join(temp, `native-windows-controller-io-${suffix}`),
      {kind: "io", handoffSha256: pins.ioHandoff}),
    windowsHermes = deps.windowsControllerHandoffInputs(hermesRoot, hermesExpectation),
    dpcppComponentRoot = path.join("/tmp", `native-windows-dpcpp-static-${suffix}`),
    hostComponentRoot = path.join("/tmp", `native-windows-host-component-${suffix}`),
    dpcppComponent = deps.windowsDpcppComponentInputs(dpcppComponentRoot), hostComponent = deps.windowsHostComponentInputs(hostComponentRoot),
    hipDriverLibrary = hostComponent.runtimeInputs["amdhip64"] || "";
  if (dpcppComponent.files[path.join(dpcppComponentRoot, "component.json")] !== pins.dpcppComponentManifest ||
      hostComponent.files[path.join(hostComponentRoot, "component.json")] !== pins.hostComponentManifest ||
      fileHash(path.join(dpcppComponentRoot, "component.json")) !== pins.dpcppComponentManifest ||
      fileHash(path.join(hostComponentRoot, "component.json")) !== pins.hostComponentManifest ||
      io.kind !== "io" || windowsHermes.kind !== "hermes" || !windowsHermes.intelRuntime || !hipDriverLibrary) {
    throw new Error("Windows application component or support manifest differs from its external pin");
  }
  const hermesInputs = deps.nativeHermesInputs(path.join(hermes.sdkRoot, "result.json")),
    toolsParent = path.join(temp, `native-linux-llvm-tools-${suffix}`),
    tools = deps.linuxLlvmToolHandoffInputs(toolsParent, {handoffSha256: pins.linuxToolsHandoff}),
    hermesCompiler = hermesInputs.artifacts["compiler"] || "";
  if (!hermesCompiler || hermesInputs.files[hermesCompiler] !== fileHash(hermesCompiler) ||
      tools.toolRoot !== path.join(toolsParent, "bin")) {throw new Error("Linux Hermes AOT or thin LLVM tools did not re-admit");}
  const imagesState = ci.cudaImagesStatePaths(paths, environment),
    descriptorState = /** @type {Parameters<BuildDependencies["ci"]["auditLinuxNativeSdkDescriptorCudaImages"]>[0]} */ (
      /** @type {unknown} */ ({images: imagesState})),
    imageAdmission = ci.auditLinuxNativeSdkDescriptorCudaImages(descriptorState, dpcppInputs),
    imagePath = imageAdmission.include, cudaInclude = path.join(dpcpp.cudaRoot, "include"), hipInclude = hipHeaders.include,
    nvmlHeader = path.join(cudaInclude, "nvml.h");
  if (imagePath !== path.join(imagesState.handoffRoot, "cuda-images.inc") ||
      !imageAdmission.includeSha256 || !fs.lstatSync(nvmlHeader).isFile() || !fs.lstatSync(path.join(cudaInclude, "cuda.h")).isFile() ||
      !fs.lstatSync(path.join(hipInclude, "hip/hip_runtime_api.h")).isFile() ||
      !/\.lib$/i.test(hipDriverLibrary) || !fs.lstatSync(hipDriverLibrary).isFile()) {
    throw new Error("Windows application did not admit the exact CUDA, HIP or embedded-image include roots");
  }
  const hostSdkRoot = windowsHostSdkPaths(temp, suffix).sdkOutput,
    sdkPins = deps.windowsGpu.windowsGpuHostSdkPins(hostSdkRoot, ci.buildWindowsHostSdkSourcesCi()),
    toolRoot = tools.toolRoot, clangCl = path.join(intelRoot, "bin/icx-cl"),
    llvmDescriptor = ci.composeWindowsLlvmSdkDescriptor({windowsSdkRoot: hostSdkRoot, windowsSdkPins: sdkPins, hermesRoot, hermesExpectation,
      tools: {clangCl: {path: clangCl, sha256: fileHash(clangCl)},
        lldLink: {path: tools.lldLink, sha256: fileHash(tools.lldLink), linkTarget: "lld"},
        llvmNm: {path: tools.llvmNm, sha256: fileHash(tools.llvmNm)}, clangResourceInclude: path.join(intelRoot, "lib/clang/23/include")}}),
    application = {core: core.archive, gpuReceipts, gpuHandoffs, defaultCompiler: "dpcpp",
      tools: {llvmAr: path.join(toolRoot, "llvm-ar"), llvmObjcopy: path.join(toolRoot, "llvm-objcopy")},
      sdk: {libraries: [], wholeArchives: [], objects: [], includeRoots: [cudaInclude, hipInclude], nvmlHeader,
        hipDriverLibrary, dpcppComponent: dpcppComponentRoot, hostComponent: hostComponentRoot, cudaImages: imagePath}},
    config = deps.composeWindowsApplicationConfig({llvmDescriptor: llvmDescriptor.descriptor, io,
      hermes: windowsHermes, hermesAot: hermesInputs, tools: {shermaot: hermesCompiler,
        llvmReadobj: path.join(toolRoot, "llvm-readobj")}, allowedImports: deps.windowsSystemImports(), application}),
    configText = JSON.stringify(config, null, 2) + "\n";
  fs.writeFileSync(plan.configFile, configText, {flag: "wx", mode: 0o444});
  if (fileHash(plan.configFile) !== crypto.createHash("sha256").update(configText).digest("hex")) {
    throw new Error("Composed Windows application descriptor changed during admission");
  }
  return {hipRoot, hipExpectation, hipHeaders, toolRoot};
}

/** Run the public release command inside the host memory scope and the aggregate deadline.
 * Admission pins and the aggregate deadline belong to transport, not compilation: the
 * descriptor binds actual inputs and the outer timeout keeps the deadline without poisoning
 * every object-cache key.
 * @param {NodeJS.ProcessEnv} environment @param {string} toolRoot @param {string} configFile @param {string} output
 * @param {number} deadline @param {BuildDependencies} deps */
function runReleaseBuild(environment, toolRoot, configFile, output, deadline, deps) {
  /** @type {NodeJS.ProcessEnv} */
  const buildEnvironment = {...environment, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "4",
    PATH: [path.join(intelRoot, "bin"), toolRoot, environment["PATH"] || "", "/usr/local/bin", "/usr/bin", "/bin"].filter(Boolean).join(":"),
    LD_LIBRARY_PATH: [path.join(intelRoot, "lib"), "/usr/lib/x86_64-linux-gnu"].join(":")};
  for (const key of Object.keys(buildEnvironment)) {
    if (key.startsWith("MOM_NATIVE_WINDOWS_APPLICATION_") || key === "MOM_CI_DEADLINE_EPOCH") {delete buildEnvironment[key];}
  }
  const timeoutMs = remainingMs(deadline, 45 * 60 * 1000, 5 * 60 * 1000), seconds = Math.floor(timeoutMs / 1000),
    timeout = fs.realpathSync("/usr/bin/timeout"), memoryScope = path.join(repo, "scripts/run-bounded-native-ci.sh");
  const build = deps.spawnSync(memoryScope, [timeout, "--signal=TERM", "--kill-after=5s", `${seconds}s`, "npm", "run", "build:release", "--",
    "--native", configFile, output], {cwd: repo, env: buildEnvironment, stdio: "inherit", timeout: timeoutMs + 10000});
  if (build.error || build.signal || build.status !== 0) {throw new Error("Bounded Windows native application build failed", {cause: build.error});}
}

/** @param {fs.Stats} stat */
function boundedReceipt(stat) {
  return stat.isFile() && !stat.isSymbolicLink() && stat.nlink === 1 && stat.size >= 1 && stat.size <= 128 * MiB;
}

/** Prove the producer receipt binds the raw build-only executable at its planned path.
 * @param {string} output @param {string} configFile @param {BuildDependencies} deps */
function admitBuildOutput(output, configFile, deps) {
  const {fileHash, readBuildJsonSnapshot} = deps.sourceInputs, executable = path.join(output, "mom.exe"),
    resultFile = path.join(output, "result.json"), entrypoint = path.join(output, "entrypoint.json"),
    resultIdentity = fs.lstatSync(resultFile), entrypointIdentity = fs.lstatSync(entrypoint);
  if (!boundedReceipt(resultIdentity) || !boundedReceipt(entrypointIdentity)) {
    throw new Error("Windows application result and entrypoint receipts are not bounded canonical files");
  }
  const {value: receipt, sha256: resultSha256} = readBuildJsonSnapshot(resultFile), entrypointSha256 = fileHash(entrypoint),
    stat = fs.lstatSync(executable);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size >= 1 << 30 ||
      fs.realpathSync(executable) !== executable || receipt.target !== "x86_64-pc-windows-msvc" || receipt.nativeRuntime !== true ||
      receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
      receipt.configHash !== fileHash(configFile) || receipt.artifact?.path !== executable || receipt.artifact.bytes !== stat.size ||
      receipt.artifact.sha256 !== fileHash(executable) || !receipt.application || receipt.application.graphs?.length !== profiles.length ||
      fileHash(resultFile) !== resultSha256 || fileHash(entrypoint) !== entrypointSha256 ||
      resultIdentity.dev !== fs.lstatSync(resultFile).dev || resultIdentity.ino !== fs.lstatSync(resultFile).ino ||
      entrypointIdentity.dev !== fs.lstatSync(entrypoint).dev || entrypointIdentity.ino !== fs.lstatSync(entrypoint).ino) {
    throw new Error("Windows application producer receipt does not bind the raw build-only executable");
  }
  return {executable, resultFile, entrypoint};
}

/** `build` command: restore the pinned cohort from RUNNER_TEMP/incoming, compose the
 * application descriptor, link mom.exe, and publish the planned paths for `seal`.
 * In CI `environment` is process.env, so the host-SDK settings reach the SDK producer.
 * @param {NodeJS.ProcessEnv} [environment] @param {BuildDependencies} [deps] */
function runBuild(environment = process.env, deps = realBuildDependencies()) {
  const {ci} = deps, paths = ci.ciPaths(), suffix = paths.suffix, temp = paths.runnerTemp, deadline = ciDeadline(environment),
    output = path.join(temp, `native-windows-application-output-${suffix}`),
    configFile = path.join(temp, `native-windows-application-config-${suffix}.json`);
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== ciBuildJobs(environment) ||
      fs.realpathSync(temp) !== temp || fs.existsSync(output) || fs.existsSync(configFile)) {
    throw new Error("Windows application inputs or fresh output paths differ from the bounded run plan");
  }
  const {pins, gpuPins} = admitPins(environment);
  Object.assign(environment, windowsHostSdkEnvironment(temp, suffix, deadline));
  const rows = inputRows(path.join(temp, "incoming"), temp, suffix, pins, gpuPins, {hermes: ci.hermesStatePaths(paths, environment),
      dpcpp: ci.dpcppCudaInputsStatePaths(paths, environment), linuxSource: ci.linuxSourceStatePaths(paths, environment)}),
    inputs = pinInputs(rows, deadline, deps);
  assertAggregateSpace(inputs);
  for (const input of inputs) {restoreInput(input, deadline, deps);}
  releaseTransports(inputs, deps);
  const {hipRoot, hipExpectation, hipHeaders, toolRoot} = composeApplication({suffix, temp, configFile, pins, gpuPins, paths, environment}, deps);
  runReleaseBuild(environment, toolRoot, configFile, output, deadline, deps);
  if (!isDeepStrictEqual(hipHeaders.files, deps.rocmHipHeaderInputs(hipRoot, hipExpectation).files)) {
    throw new Error("HIP header files changed during the Windows application build");
  }
  const {executable, resultFile, entrypoint} = admitBuildOutput(output, configFile, deps);
  fs.appendFileSync(environment["GITHUB_OUTPUT"] || "", [
    `build_output=${output}`, `config_file=${configFile}`, `executable=${executable}`,
    `result_file=${resultFile}`, `entrypoint_file=${entrypoint}`,
  ].join("\n") + "\n");
}

/** @param {string} file @param {string} label @param {number} maximum */
function canonicalRegular(file, label, maximum) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > maximum ||
      fs.realpathSync(file) !== file || path.resolve(file) !== file) {throw new Error("Invalid Windows application " + label);}
  return {bytes: stat.size, sha256: fileHash(file), identity: fileIdentity(stat)};
}

/** The composed config must carry exactly one same-run receipt pin per maintained GPU profile.
 * @param {unknown} gpuPins */
function sealedGpuPinsValid(gpuPins) {
  return Array.isArray(gpuPins) && gpuPins.length === profiles.length && gpuPins.every(pin =>
    pin && typeof pin === "object" && !Array.isArray(pin) && hasExactBuildFields(pin, sealedGpuPinKeys) &&
    profiles.includes(pin.profile) &&
    [pin.configSha256, pin.handoffSha256, pin.receiptSha256].every(value => typeof value === "string" && sha256Pattern.test(value))) &&
    new Set(gpuPins.map(pin => pin.profile)).size === profiles.length;
}

/** Cross-check config, producer receipt, entrypoint receipt and executable before sealing.
 * @param {any} config @param {any} result @param {any} entrypoint @param {{executable:string,configFile:string,resultFile:string}} plan
 * @param {{configSha256:string,resultSha256:string,artifact:{bytes:number,sha256:string}}} records */
function sealedReceiptsValid(config, result, entrypoint, plan, records) {
  const {executable, configFile, resultFile} = plan, {artifact} = records;
  return Boolean(config.application) && sealedGpuPinsValid(config.application.gpuHandoffs) &&
    result.target === "x86_64-pc-windows-msvc" && result.nativeRuntime === true && result.buildOnly === true &&
    result.executed === false && result.publicationQualified === false && result.configHash === records.configSha256 &&
    result.artifact?.path === executable && result.artifact.bytes === artifact.bytes && result.artifact.sha256 === artifact.sha256 &&
    Array.isArray(result.application?.graphs) && result.application.graphs.length === profiles.length &&
    new Set(result.application.graphs.map((/** @type {any} */ row) => row.factory)).size === profiles.length &&
    result.application.graphs.every((/** @type {any} */ row) =>
      profiles.includes(String(row.factory).replace("mom_native_gpu_", "").replaceAll("_", "-"))) &&
    entrypoint.version === 1 && entrypoint.target === "win32" && entrypoint.configFile === configFile &&
    entrypoint.configHash === records.configSha256 && entrypoint.source?.path === executable && entrypoint.source.sha256 === artifact.sha256 &&
    entrypoint.artifact?.path === executable && entrypoint.artifact.sha256 === artifact.sha256 && entrypoint.artifact.bytes === artifact.bytes &&
    entrypoint.producerReceipt?.path === resultFile && entrypoint.producerReceipt.sha256 === records.resultSha256 &&
    entrypoint.buildOnly === true && entrypoint.publicationQualified === false;
}

/** `seal` command: copy the config and both receipts into a compact evidence tar beside a
 * handoff.json, re-prove that nothing changed, and publish the artifact names and digests.
 * @param {NodeJS.ProcessEnv} [environment] */
function runSeal(environment = process.env) {
  const ci = require("../native-ci-tar"), suffix = ciSuffix(environment), temp = environment["RUNNER_TEMP"] || "",
    buildRoot = path.join(temp, `native-windows-application-output-${suffix}`),
    configFile = path.join(temp, `native-windows-application-config-${suffix}.json`),
    executable = path.join(buildRoot, "mom.exe"), resultFile = path.join(buildRoot, "result.json"),
    entrypointFile = path.join(buildRoot, "entrypoint.json"), evidenceName = applicationBuildName("windows", suffix),
    evidenceRoot = path.join("/tmp", evidenceName), archive = path.join(temp, evidenceName + ".tar"), output = environment["GITHUB_OUTPUT"] || "";
  const deadline = ciDeadline(environment);
  if (fs.realpathSync(temp) !== temp ||
      environment["BUILD_OUTPUT"] !== buildRoot || environment["CONFIG_FILE"] !== configFile || environment["EXECUTABLE"] !== executable ||
      environment["RESULT_FILE"] !== resultFile || environment["ENTRYPOINT_FILE"] !== entrypointFile ||
      fs.existsSync(evidenceRoot) || fs.existsSync(archive)) {throw new Error("Windows application seal paths differ from the CI work-root plan");}
  const {value: config, sha256: configSha256} = readCanonicalBuildJson(configFile, 8 << 20, "Windows application composed config"),
    {value: result, sha256: resultSha256} = readCanonicalBuildJson(resultFile, 8 << 20, "Windows application producer receipt"),
    {value: entrypoint, sha256: entrypointSha256} = readCanonicalBuildJson(entrypointFile, 8 << 20, "Windows application entrypoint receipt"),
    artifact = canonicalRegular(executable, "runtime executable", (1 << 30) - 1);
  if (!sealedReceiptsValid(config, result, entrypoint, {executable, configFile, resultFile},
    {configSha256, resultSha256, artifact}) || fs.existsSync(evidenceRoot)) {
    throw new Error("Windows application receipts, same-run GPU cohort or runtime executable failed sealing");
  }
  fs.mkdirSync(evidenceRoot, {mode: 0o700});
  /** @type {Array<[string, string, string]>} */
  const evidence = [[configFile, "application-config.json", configSha256],
    [resultFile, "result.json", resultSha256], [entrypointFile, "entrypoint.json", entrypointSha256]];
  for (const [source, name, sha256] of evidence) {
    const destination = path.join(evidenceRoot, name);
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
    if (fileHash(destination) !== sha256 || fileHash(source) !== sha256) {throw new Error("Windows application build evidence copy changed: " + name);}
  }
  const handoff = {version: 1, format: "mom-windows-application-build-handoff-v1",
      target: "x86_64-pc-windows-msvc", artifact: {file: "mom.exe", bytes: artifact.bytes, sha256: artifact.sha256},
      config: {file: "application-config.json", sha256: configSha256},
      producerReceipt: {file: "result.json", sha256: resultSha256},
      entrypointReceipt: {file: "entrypoint.json", sha256: entrypointSha256},
      gpuHandoffs: config.application.gpuHandoffs, buildOnly: true, executed: false, publicationQualified: false},
    handoffFile = path.join(evidenceRoot, "handoff.json");
  fs.writeFileSync(handoffFile, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  const handoffSha256 = fileHash(handoffFile), packed = ci.packState(archive, "/tmp", [evidenceName], remainingMs(deadline, 5 * 60 * 1000, 0)),
    expectedMembers = [evidenceName, ...applicationBuildFiles
      .map(name => `${evidenceName}/${name}`)].sort(),
    members = ci.listState(archive, [evidenceName], remainingMs(deadline, 5 * 60 * 1000, 0)).map(name => name.replace(/\/$/, "")).sort();
  if (packed.bytes < 1 || packed.bytes > 128 << 20 || !sha256Pattern.test(handoffSha256) ||
      members.join("\0") !== expectedMembers.join("\0") || fileHash(handoffFile) !== handoffSha256 ||
      artifact.identity.dev !== fs.lstatSync(executable).dev || artifact.identity.ino !== fs.lstatSync(executable).ino ||
      fileHash(executable) !== artifact.sha256 || fileHash(configFile) !== configSha256 ||
      fileHash(resultFile) !== resultSha256 || fileHash(entrypointFile) !== entrypointSha256) {
    throw new Error("Windows application runtime or compact build handoff changed during sealing");
  }
  const bundleSha256 = fileHash(archive);
  assertWorkflowFile(output);
  fs.appendFileSync(output, [
    `executable_artifact_name=native-windows-executable-${suffix}`,
    `executable_sha256=${artifact.sha256}`, `executable_bytes=${artifact.bytes}`,
    `build_handoff_artifact_name=${evidenceName}.tar`, `build_handoff_bundle_sha256=${bundleSha256}`,
    `build_handoff_sha256=${handoffSha256}`,
  ].join("\n") + "\n");
}

module.exports = {profiles, admitPins, ciDeadline, inputRows, runPins, runBuild, runSeal};

if (require.main === module) {
  /** @type {Record<string, () => void>} */
  const commands = {pins: () => runPins(), build: () => runBuild(), seal: () => runSeal()};
  const command = commands[process.argv[2] || ""];
  try {
    if (!command || process.argv.length !== 3) {throw new Error("Usage: windows-application.js pins|build|seal");}
    command();
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
