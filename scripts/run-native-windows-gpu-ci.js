"use strict";

// One Windows factory job reuses the original source/input producers and GPU builder.
// Downloaded tars are retired only after successful pinned restoration and admission.
const fs = require("node:fs"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const ci = require("./prepare-native-ci"), {fileHash, readBuildJson} = require("./build-source-inputs");
const {sdkComponents} = require("./prepare-native-sdk-sources");
const {linuxLlvmToolHandoffInputs, linuxLlvmAcppFrontendInputs} = require("./native-linux-llvm-tools");
const {windowsNativeGpuConfig, windowsNativeAcppGpuConfig} = require("./native-windows-gpu-config");
// Reserve the producer's five-minute build minimum, seven-minute pack budget and admission margin.
const factoryReserveMs = 14 * 60 * 1000;
const windowsGpuProfiles = Object.freeze(["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]);
const acppGpuProfiles = Object.freeze(/** @type {string[]} */ (["acpp-cuda", "acpp-hip"]));

/** Keep ACPP plugin/frontend execution free of unrelated DPC++ or Intel loader state. @param {string} profile @param {string|null} dpcppRoot */
function windowsGpuLibraryPath(profile, dpcppRoot) {
  if (profile === "oneapi") {return ["/opt/intel/oneapi/compiler/2026.2/lib", "/usr/lib/x86_64-linux-gnu"].join(":");}
  if (profile === "dpcpp" || profile === "dpcpp-opencl") {
    if (!dpcppRoot) {throw new Error("DPC++ profiles require their admitted compiler root");}
    return [path.join(dpcppRoot, "lib"), "/usr/lib/x86_64-linux-gnu"].join(":");
  }
  if (acppGpuProfiles.includes(profile)) {return "/usr/lib/x86_64-linux-gnu";}
  throw new Error("Unsupported Windows GPU profile loader environment");
}

/** Build the exact ACPP input object passed to the public config composer.
 * @param {ReturnType<typeof windowsGpuCiJobPlan>} plan @param {string} windowsSdkRoot
 * @param {ReturnType<typeof windowsGpuHostSdkPins>} windowsSdkPins */
function windowsGpuAcppConfigInput(plan, windowsSdkRoot, windowsSdkPins) {
  const {acpp, dpcpp, cudaImages, hipHeaders, pins} = plan;
  if (!acpp) {throw new Error("The selected Windows GPU profile has no admitted ACPP inputs");}
  return {windowsSdkRoot, windowsSdkPins, toolsRoot: plan.toolsRoot,
    toolsExpectation: {runAttempt: plan.suffix, handoffSha256: pins["TOOLS_HANDOFF"] || ""},
    frontendRoot: acpp.frontendRoot,
    frontendExpectation: {runAttempt: plan.suffix, handoffSha256: pins["FRONTEND_HANDOFF"] || ""},
    acpp: {archive: acpp.sdkTar, bundleSha256: pins["ACPP_SDK_BUNDLE"] || "", root: acpp.sdkRoot,
      sourceRoot: acpp.source.preparedCheckout},
    ...(dpcpp ? {dpcppState: dpcpp, dpcppManifestSha256: pins["DPCPP_MANIFEST"] || ""} : {}),
    ...(cudaImages ? {cudaImages} : {}),
    ...(hipHeaders ? {hipHeaders: {root: hipHeaders.root, handoffSha256: pins["HIP_HEADERS_HANDOFF"] || ""}} : {})};
}

/** Validate caller pins and derive only the maintained same-run paths; no mutation or compiler queries.
 * @param {string} profile */
function windowsGpuCiJobPlan(profile) {
  const environment = process.env;
  if (!windowsGpuProfiles.includes(profile) || environment["MOM_BUILD_JOBS"] !== "2") {
    throw new Error("Windows GPU CI stages only the five maintained two-job factories");
  }
  const paths = ci.ciPaths(), linux = ci.linuxSourceStatePaths(paths, environment),
    dpcpp = profile === "acpp-hip" ? null : ci.dpcppCudaInputsStatePaths(paths, environment),
    acppSource = acppGpuProfiles.includes(profile) ? ci.acppStatePaths(paths, environment) : null,
    factory = ci.windowsGpuHandoffStatePaths(profile, environment), suffix = paths.suffix,
    pinNames = ["SOURCE_BUNDLE", "TOOLS_BUNDLE", "TOOLS_HANDOFF",
      ...(dpcpp ? ["DPCPP_BUNDLE", "DPCPP_MANIFEST"] : []),
      ...(profile === "acpp-hip" ? ["HIP_HEADERS_BUNDLE", "HIP_HEADERS_HANDOFF"] : []),
      ...(acppSource ? ["ACPP_SOURCE_BUNDLE", "ACPP_SDK_BUNDLE", "FRONTEND_BUNDLE", "FRONTEND_HANDOFF"] : [])],
    pins = Object.fromEntries(pinNames.map(name => [name, environment["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"] || ""]));
  if (Object.values(pins).some(pin => !/^[a-f0-9]{64}$/.test(pin))) {
    throw new Error("Windows GPU CI requires every selected input's external same-run SHA-256 pin");
  }
  const imagesPin = environment["MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256"] || "";
  const needsImages = profile === "dpcpp" || profile === "acpp-cuda";
  if (needsImages ? !/^[a-f0-9]{64}$/.test(imagesPin) : imagesPin !== "") {
    throw new Error("Only the DPC++ CUDA or ACPP CUDA factories require an external image-bundle SHA-256");
  }
  const acppPinNames = ["ACPP_SOURCE_BUNDLE", "ACPP_SDK_BUNDLE", "FRONTEND_BUNDLE", "FRONTEND_HANDOFF"],
    dpcppPinNames = ["DPCPP_BUNDLE", "DPCPP_MANIFEST"];
  if ((!acppSource && acppPinNames.some(name => environment["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"])) ||
      (profile !== "acpp-hip" && ["HIP_HEADERS_BUNDLE", "HIP_HEADERS_HANDOFF"].some(name => environment["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"])) ||
      (!dpcpp && dpcppPinNames.some(name => environment["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"]))) {
    throw new Error("Windows GPU job received an input pin unused by its selected profile");
  }
  const deadlineEpoch = environment["MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"] || "";
  ci.windowsGpuDeadlineBudget(deadlineEpoch, "build");
  const toolsName = "native-linux-llvm-tools-" + suffix, incoming = path.join(paths.runnerTemp, "incoming"),
    toolsParent = path.join(paths.runnerTemp, "windows-gpu-tools-" + suffix),
    configSource = path.join(paths.runnerTemp, "windows-gpu-" + profile + "-config-" + suffix + ".json"),
    windowsRootsFile = path.join(paths.runnerTemp, "windows-gpu-" + profile + "-source-roots-" + suffix + ".json"),
    sdkBase = path.join(paths.runnerTemp, "mom-native-windows-host-sdk-" + suffix),
    sdkMetadata = path.join(paths.runnerTemp, "native-windows-host-sdk-" + suffix + "-metadata.tar"),
    imagesName = "mom-native-cuda-images-handoff-" + suffix,
    cudaImages = needsImages ? {archive: path.join(incoming, "native-linux-cuda-images-" + suffix + ".tar"),
      bundleSha256: imagesPin, root: path.join("/tmp", imagesName)} : null;
  const hipHeadersName = "native-rocm-hip-headers-" + suffix,
    hipHeaders = profile === "acpp-hip" ? {name: hipHeadersName, root: path.join("/tmp", hipHeadersName),
      archive: path.join(incoming, hipHeadersName + ".tar")} : null;
  const acpp = acppSource ? {source: acppSource,
    sourceTar: path.join(incoming, path.basename(acppSource.sourceTar)),
    sdkName: "mom-native-acpp-cuda-hip-sdk-" + suffix,
    sdkRoot: path.join("/tmp", "mom-native-acpp-cuda-hip-sdk-" + suffix),
    sdkTar: path.join(incoming, "native-linux-acpp-cuda-hip-sdk-" + suffix + ".tar"),
    frontendName: "native-linux-llvm-acpp-frontend-" + suffix,
    frontendRoot: path.join("/tmp", "native-linux-llvm-acpp-frontend-" + suffix),
    frontendTar: path.join(incoming, "native-linux-llvm-acpp-frontend-" + suffix + ".tar")} : null;
  return {paths, linux, dpcpp, acpp, hipHeaders, factory, suffix, pins, deadlineEpoch, toolsName, toolsParent,
    toolsRoot: path.join(toolsParent, toolsName), sourceTar: linux.incomingTar,
    dpcppTar: dpcpp ? path.join(incoming, path.basename(dpcpp.sdkTar)) : null, toolsTar: path.join(incoming, toolsName + ".tar"),
    configSource, windowsRootsFile, sdkBase, sdkMetadata, cudaImages, imagesName};
}

/** Reject path/report aliases and stale outputs before restoring or retiring any input.
 * @param {string} profile */
function windowsGpuCiJobPreflight(profile) {
  const plan = windowsGpuCiJobPlan(profile), reports = [process.env["GITHUB_OUTPUT"] || "", process.env["GITHUB_STEP_SUMMARY"] || ""],
    fresh = [plan.configSource, plan.windowsRootsFile, plan.toolsParent, plan.factory.root,
      plan.linux.source, plan.linux.prepared, ...(plan.dpcpp ? [plan.dpcpp.root] : []),
      ...(plan.acpp ? [plan.acpp.source.source, plan.acpp.sdkRoot, plan.acpp.frontendRoot] : []),
      ...(plan.hipHeaders ? [plan.hipHeaders.root] : []),
      plan.sdkBase, plan.sdkMetadata, plan.factory.archive,
      ...(plan.cudaImages ? [plan.cudaImages.root] : [])];
  if (fs.realpathSync("/tmp") !== "/tmp" || fresh.some(file => fs.existsSync(file))) {
    throw new Error("Windows GPU CI requires fresh canonical input/output roots");
  }
  const reportIds = reports.map(file => {
    if (!path.isAbsolute(file) || /[\r\n\0]/.test(file)) {throw new Error("Windows GPU workflow reports must be canonical empty files");}
    const stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size !== 0 || fs.realpathSync(file) !== file ||
        fresh.some(root => file === root || file.startsWith(root + path.sep))) {
      throw new Error("Windows GPU workflow reports must be canonical empty files outside artifact roots");
    }
    return `${stat.dev}:${stat.ino}`;
  });
  if (new Set(reportIds).size !== reports.length) {throw new Error("Windows GPU workflow reports alias one another");}
  const archives = /** @type {Array<[string,string]>} */ ([[plan.sourceTar, plan.pins["SOURCE_BUNDLE"] || ""],
    [plan.toolsTar, plan.pins["TOOLS_BUNDLE"] || ""],
    ...(plan.dpcpp ? [[plan.dpcppTar || "", plan.pins["DPCPP_BUNDLE"] || ""]] : []),
    ...(plan.acpp ? [[plan.acpp.sourceTar, plan.pins["ACPP_SOURCE_BUNDLE"] || ""],
      [plan.acpp.sdkTar, plan.pins["ACPP_SDK_BUNDLE"] || ""], [plan.acpp.frontendTar, plan.pins["FRONTEND_BUNDLE"] || ""]] : [])]);
  if (plan.hipHeaders) {archives.push([plan.hipHeaders.archive, plan.pins["HIP_HEADERS_BUNDLE"] || ""]);}
  for (const [file, pin] of archives) {pinnedArchive(file, pin);}
  if (plan.cudaImages) {pinnedArchive(plan.cudaImages.archive, plan.cudaImages.bundleSha256);}
  return plan;
}

/** @param {string} file @param {string} expected */
function pinnedArchive(file, expected) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
      /[\r\n\0]/.test(file) || stat.size < 1 || stat.size > 16 * 1024 ** 3 || fileHash(file) !== expected) {
    throw new Error("Windows GPU input archive differs from its external pin: " + path.basename(file));
  }
}

/** Keep every subprocess and restore within the original job deadline.
 * @param {ReturnType<typeof windowsGpuCiJobPlan>} plan @param {number} capMs @param {number} [reserveMs] */
function remaining(plan, capMs, reserveMs = 60000) {
  const ms = Math.min(capMs, Number(plan.deadlineEpoch) * 1000 - Date.now() - reserveMs);
  if (!Number.isSafeInteger(ms) || ms < 1000) {throw new Error("Windows GPU input preparation exhausted its job deadline");}
  return ms;
}

/** Bind SDK metadata to the just-finished producer, not a new self-attested hash.
 * @param {string} root @param {Pick<ReturnType<typeof ci.buildWindowsHostSdkSourcesCi>,
 * "sdk_receipt_sha256"|"descriptor_sha256"|"files_provenance_sha256">} producer */
function windowsGpuHostSdkPins(root, producer) {
  const resultFile = path.join(root, "result.json");
  if (fileHash(resultFile) !== producer.sdk_receipt_sha256) {throw new Error("Windows GPU SDK receipt changed after its producer");}
  const result = readBuildJson(resultFile);
  if (result.descriptor?.sha256 !== producer.descriptor_sha256 || result.provenance?.sha256 !== producer.files_provenance_sha256 ||
      !/^[a-f0-9]{64}$/.test(result.overlay?.sha256 || "")) {
    throw new Error("Windows GPU SDK metadata differs from its producer pins");
  }
  return {resultSha256: producer.sdk_receipt_sha256, descriptorSha256: producer.descriptor_sha256,
    filesSha256: producer.files_provenance_sha256, overlaySha256: result.overlay.sha256};
}

/** A same-run DPC++ producer must have consumed this exact source artifact, not another tar.
 * @param {string} file @param {string} manifestSha256 @param {string} sourceSha256 */
function windowsGpuDpcppManifest(file, manifestSha256, sourceSha256) {
  if (![manifestSha256, sourceSha256].every(pin => /^[a-f0-9]{64}$/.test(pin)) || fileHash(file) !== manifestSha256) {
    throw new Error("DPC++ input manifest differs from its external pin");
  }
  const manifest = readBuildJson(file);
  if (manifest.source?.tarSha256 !== sourceSha256) {throw new Error("DPC++ inputs consumed a different source artifact");}
  return manifest;
}

/** Re-admit the restored same-run DPC++ bundle with its already verified manifest pin.
 * The caller must first verify the external bundle pin and restore this root from that tar.
 * @param {ReturnType<typeof ci.dpcppCudaInputsStatePaths>} state @param {string} manifestSha256 @param {string} sourceSha256 */
function admitWindowsGpuDpcppInputs(state, manifestSha256, sourceSha256) {
  const manifest = windowsGpuDpcppManifest(state.manifest, manifestSha256, sourceSha256);
  return ci.auditDpcppCudaInputs(state, manifest, manifestSha256);
}

/** Restore one exact top-level cohort and keep a four-GiB disk reserve without keeping two copies.
 * @param {ReturnType<typeof windowsGpuCiJobPlan>} plan @param {string} archive @param {string} pin
 * @param {string} destination @param {string} root */
function restore(plan, archive, pin, destination, root) {
  pinnedArchive(archive, pin);
  const members = ci.listState(archive, [root], remaining(plan, 5 * 60 * 1000)),
    expanded = ci.tarExpandedBytes(archive, members.length, remaining(plan, 5 * 60 * 1000)),
    space = fs.statfsSync(destination, {bigint: true});
  if (BigInt(expanded) + 4n * 1024n ** 3n > space.bavail * space.bsize) {
    throw new Error("Windows GPU input restore would consume its four-GiB disk reserve");
  }
  ci.extractState(archive, destination, [root], [root], remaining(plan, 10 * 60 * 1000));
}

/** Run one existing factory producer; no runtime or hardware test is performed.
 * @param {string} profile */
function runWindowsGpuCiJob(profile) {
  const plan = windowsGpuCiJobPreflight(profile), {paths, linux, dpcpp, acpp, pins} = plan, git = fs.realpathSync("/usr/bin/git");
  restore(plan, plan.sourceTar, pins["SOURCE_BUNDLE"] || "", "/tmp", linux.sourceName);
  ci.verifyLinuxSources(paths, linux, git);
  fs.unlinkSync(plan.sourceTar);
  if (dpcpp) {
    restore(plan, plan.dpcppTar || "", pins["DPCPP_BUNDLE"] || "", "/tmp", dpcpp.name);
    admitWindowsGpuDpcppInputs(dpcpp, pins["DPCPP_MANIFEST"] || "", pins["SOURCE_BUNDLE"] || "");
    fs.unlinkSync(plan.dpcppTar || "");
  }
  fs.mkdirSync(plan.toolsParent, {mode: 0o700});
  restore(plan, plan.toolsTar, pins["TOOLS_BUNDLE"] || "", plan.toolsParent, plan.toolsName);
  linuxLlvmToolHandoffInputs(plan.toolsRoot, {runAttempt: plan.suffix, handoffSha256: pins["TOOLS_HANDOFF"] || ""});
  fs.unlinkSync(plan.toolsTar);
  if (acpp) {
    restore(plan, acpp.sourceTar, pins["ACPP_SOURCE_BUNDLE"] || "", "/tmp", acpp.source.sourceName);
    ci.verifyAcppSource(paths, acpp.source, git);
    fs.unlinkSync(acpp.sourceTar);
    restore(plan, acpp.frontendTar, pins["FRONTEND_BUNDLE"] || "", "/tmp", acpp.frontendName);
    linuxLlvmAcppFrontendInputs(acpp.frontendRoot, {runAttempt: plan.suffix, handoffSha256: pins["FRONTEND_HANDOFF"] || ""});
    fs.unlinkSync(acpp.frontendTar);
    restore(plan, acpp.sdkTar, pins["ACPP_SDK_BUNDLE"] || "", "/tmp", acpp.sdkName);
  }
  if (plan.cudaImages) {
    restore(plan, plan.cudaImages.archive, plan.cudaImages.bundleSha256, "/tmp", plan.imagesName);
  }
  if (plan.hipHeaders) {
    restore(plan, plan.hipHeaders.archive, pins["HIP_HEADERS_BUNDLE"] || "", "/tmp", plan.hipHeaders.name);
  }

  const rawRoots = readBuildJson(linux.rootsFile), sourceRoots = Object.fromEntries(Object.keys(sdkComponents("windows"))
      .map(name => [name, rawRoots[name]])), windowsSources = path.join(linux.prepared, "windows");
  fs.writeFileSync(plan.windowsRootsFile, JSON.stringify(sourceRoots) + "\n", {flag: "wx", mode: 0o444});
  // GNU timeout owns a process group, so deadline expiry also terminates Git/patch children.
  const prepareMs = remaining(plan, 10 * 60 * 1000, factoryReserveMs), prepared = spawnSync("/usr/bin/timeout",
    ["--signal=TERM", "--kill-after=5s", Math.floor(prepareMs / 1000) + "s", process.execPath,
      path.join(__dirname, "prepare-native-sdk-sources.js"), "windows", plan.windowsRootsFile, windowsSources],
    {env: paths.env, stdio: "inherit", timeout: prepareMs + 10000});
  if (prepared.error || prepared.signal || prepared.status !== 0) {
    throw new Error("Bounded Windows GPU source preparation failed", {cause: prepared.error});
  }
  remaining(plan, 25 * 60 * 1000, factoryReserveMs);
  const sdkBase = plan.sdkBase,
    sdkRoot = path.join(sdkBase, "native-windows-host-sdk-" + plan.suffix),
    sdkDeadline = Math.floor(Math.min(Date.now() + 25 * 60 * 1000,
      Number(plan.deadlineEpoch) * 1000 - factoryReserveMs) / 1000);
  Object.assign(process.env, {MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT: sdkBase,
    MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR: path.join(sdkBase, "archives"), MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: sdkRoot,
    MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR: plan.sdkMetadata,
    MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: String(sdkDeadline)});
  const sdkProducer = ci.buildWindowsHostSdkSourcesCi();
  // Match standalone compiler queries/builds without borrowing another profile's runtime.
  process.env["LD_LIBRARY_PATH"] = windowsGpuLibraryPath(profile, dpcpp?.dpcppRoot || null);
  const windowsSdkPins = windowsGpuHostSdkPins(sdkRoot, sdkProducer),
    toolsExpectation = {runAttempt: plan.suffix, handoffSha256: pins["TOOLS_HANDOFF"] || ""};
  /** @type {{configBytes:Buffer,configSha256:string}} */
  let produced;
  if (acpp) {
    produced = windowsNativeAcppGpuConfig(/** @type {"acpp-cuda"|"acpp-hip"} */ (profile),
      windowsGpuAcppConfigInput(plan, sdkRoot, windowsSdkPins));
  } else {
    if (!dpcpp) {throw new Error("The selected DPC++ factory has no admitted compiler input");}
    produced = windowsNativeGpuConfig(profile, {windowsSdkRoot: sdkRoot, windowsSdkPins,
      windowsSourcesFile: path.join(windowsSources, "sources.json"), dpcppState: dpcpp,
      dpcppManifestSha256: pins["DPCPP_MANIFEST"] || "", toolsRoot: plan.toolsRoot,
      toolsExpectation, compilerRoot: profile === "oneapi" ? "/opt/intel/oneapi/compiler/2026.2" : dpcpp?.dpcppRoot || "",
      ...(plan.cudaImages ? {cudaImages: plan.cudaImages} : {})});
  }
  // The config reader also verifies the original image artifact before it is retired.
  if (plan.cudaImages) {fs.unlinkSync(plan.cudaImages.archive);}
  if (plan.hipHeaders) {fs.unlinkSync(plan.hipHeaders.archive);}
  if (acpp) {fs.unlinkSync(acpp.sdkTar);}
  fs.writeFileSync(plan.configSource, produced.configBytes, {flag: "wx", mode: 0o444});
  ci.buildWindowsGpuProfileCi(profile, plan.configSource, produced.configSha256);
}

module.exports = {windowsGpuCiJobPlan, windowsGpuCiJobPreflight, windowsGpuHostSdkPins, windowsGpuDpcppManifest,
  admitWindowsGpuDpcppInputs, windowsGpuLibraryPath, windowsGpuAcppConfigInput};
if (require.main === module) {
  try {
    const [profile, ...extra] = process.argv.slice(2);
    if (extra.length || !profile) {throw new Error("Usage: run-native-windows-gpu-ci.js oneapi|dpcpp|dpcpp-opencl|acpp-cuda|acpp-hip");}
    runWindowsGpuCiJob(profile);
  } catch (error) {
    process.stderr.write("[native-ci] Windows GPU factory failed: " + (error instanceof Error ? error.message : String(error)) + "\n");
    process.exitCode = 1;
  }
}
