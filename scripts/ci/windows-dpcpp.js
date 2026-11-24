"use strict";
// Windows original static DPC++ component job: deadline/pin preflight, same-run input re-admission and
// producer chain, and the relocatable component seal. Run from the repository checkout on a Linux x64 runner.
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const ciSources = require("../native-ci-linux-sources"),
  ciTar = require("../native-ci-tar"),
  ciWindowsLlvm = require("../native-ci-windows-llvm"),
  dpcppInputs = require("../native-dpcpp-cuda-inputs");
const {ciSuffix} = require("../native-ci-job");
const {hasExactBuildFields, fileHash, readBuildJson, readBuildJsonSnapshot} = require("../build-source-inputs");
const {sdkComponents} = require("../prepare-native-sdk-sources");
const {linuxLlvmToolHandoffInputs} = require("../native-linux-llvm-tools");
const {windowsControllerHandoffInputs} = require("../native-windows-controller-handoff");
const {nativeHermesInputs} = require("../native-release-sdk-inputs");
const {composeWindowsApplicationConfig} = require("../native-windows-application-config");
const {windowsDpcppComponentInputs} = require("../native-windows-dpcpp-component");
const {windowsGpuHostSdkPins, windowsGpuDpcppManifest} = require("../run-native-windows-gpu-ci");

const repo = path.resolve(__dirname, "../..");
const pinPrefix = "MOM_NATIVE_WINDOWS_DPCPP_STATIC_";
const pinNames = /** @type {const} */ ({
  source: "SOURCE_BUNDLE_SHA256",
  dpcpp: "DPCPP_BUNDLE_SHA256",
  dpcppManifest: "DPCPP_MANIFEST_SHA256",
  jit: "JIT_BUNDLE_SHA256",
  tools: "TOOLS_BUNDLE_SHA256",
  toolsHandoff: "TOOLS_HANDOFF_SHA256",
  io: "IO_BUNDLE_SHA256",
  ioHandoff: "IO_HANDOFF_SHA256",
  hermes: "HERMES_BUNDLE_SHA256",
  hermesHandoff: "HERMES_HANDOFF_SHA256",
  linuxHermes: "LINUX_HERMES_BUNDLE_SHA256",
});
const intelRoot = "/opt/intel/oneapi/compiler/2026.2";
const allowedImports = ["advapi32.dll", "cfgmgr32.dll", "dbghelp.dll", "kernel32.dll", "ntdll.dll", "ole32.dll",
  "oleaut32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "version.dll"];

/** @typedef {Record<keyof typeof pinNames, string>} DpcppPins */

/** Read a required, non-empty environment value (the shell step ran under `set -u`).
 * @param {NodeJS.ProcessEnv} environment @param {string} name */
function requiredEnv(environment, name) {
  const value = environment[name];
  if (!value) {throw new Error("Windows DPC++ CI requires " + name);}
  return value;
}

/** Validate every external artifact/handoff SHA-256 pin and return them keyed by role.
 * @param {NodeJS.ProcessEnv} environment @returns {DpcppPins} */
function readPins(environment) {
  const entries = Object.entries(pinNames).map(([key, name]) => {
    const pin = environment[pinPrefix + name] || "";
    if (!/^[a-f0-9]{64}$/.test(pin)) {throw new Error("Missing or malformed external artifact/handoff SHA-256 pin");}
    return [key, pin];
  });
  return /** @type {DpcppPins} */ (Object.fromEntries(entries));
}

/** Check the job deadline and all pins before any mutation, then derive the Windows host-SDK paths and
 * deadline that the in-process host-SDK producer reads from its environment.
 * @param {NodeJS.ProcessEnv} environment @param {number} [nowSeconds] */
function jobEnvironment(environment, nowSeconds = Math.floor(Date.now() / 1000)) {
  const epoch = environment["MOM_CI_DEADLINE_EPOCH"] || "";
  if (!/^[0-9]{10}$/.test(epoch)) {throw new Error("Malformed MOM_CI_DEADLINE_EPOCH job deadline");}
  const deadline = Number(epoch);
  if (deadline <= nowSeconds) {throw new Error("The Windows DPC++ job deadline is expired");}
  const pins = readPins(environment);
  if (deadline - nowSeconds < 120) {throw new Error("Insufficient time remains before the job deadline");}
  const hostSdk = ciWindowsLlvm.windowsHostSdkEnvironment(
    requiredEnv(environment, "RUNNER_TEMP"), ciSuffix(environment), deadline);
  return {deadline, pins, hostSdk};
}

/** Milliseconds a phase may use: its cap, shortened so the reserve still fits before the deadline.
 * @param {number} deadline Unix seconds @param {number} capMs @param {number} [reserveMs] */
function remainingMs(deadline, capMs, reserveMs = 60000) {
  const ms = Math.min(capMs, deadline * 1000 - Date.now() - reserveMs);
  if (!Number.isSafeInteger(ms) || ms < 1000) {throw new Error("Job deadline expired before Windows DPC++ preparation");}
  return ms;
}

/** Run a repository Node producer under GNU timeout inside the bounded-memory scope.
 * @param {number} deadline @param {string} script @param {string[]} args @param {number} capMs */
function runNode(deadline, script, args, capMs) {
  const timeoutMs = remainingMs(deadline, capMs), seconds = Math.floor(timeoutMs / 1000), timeout = fs.realpathSync("/usr/bin/timeout");
  const memoryScope = path.join(repo, "scripts/run-bounded-native-ci.sh");
  const result = spawnSync(memoryScope, [timeout, "--signal=TERM", "--kill-after=5s", `${seconds}s`, process.execPath,
    path.join(repo, script), ...args], {cwd: repo, env: process.env, stdio: "inherit", timeout: timeoutMs + 10000});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error(`Bounded Windows DPC++ child failed: ${script}${result.status === 124 ? " (deadline)" : ""}`, {cause: result.error});
  }
}

/** Require a single-link, canonical, bounded regular file whose bytes match its external pin.
 * @param {string} file @param {string} sha256 @param {number} [maximum] */
function pinArchive(file, sha256, maximum = 16 * 1024 ** 3) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > maximum ||
      fs.realpathSync(file) !== file || !/^[a-f0-9]{64}$/.test(sha256) || fileHash(file) !== sha256) {
    throw new Error("Windows DPC++ input archive differs from its external SHA-256 pin: " + path.basename(file));
  }
}

/** Extract a pinned state tar under base after member, expanded-size and four-GiB disk-reserve checks.
 * @param {number} deadline @param {string} archive @param {string} sha256 @param {string[]} roots
 * @param {string} base @param {number} maximumExpandedBytes */
function restore(deadline, archive, sha256, roots, base, maximumExpandedBytes) {
  pinArchive(archive, sha256);
  const members = ciTar.listState(archive, roots, remainingMs(deadline, 5 * 60 * 1000)),
    expanded = ciTar.tarExpandedBytes(archive, members.length, remainingMs(deadline, 5 * 60 * 1000)),
    disk = fs.statfsSync(base, {bigint: true});
  if (expanded > maximumExpandedBytes || BigInt(expanded) + 4n * 1024n ** 3n > disk.bavail * disk.bsize) {
    throw new Error("Pinned Windows DPC++ input exceeds its extraction bound or four-GiB disk reserve");
  }
  ciTar.extractState(archive, base, roots, roots, remainingMs(deadline, 10 * 60 * 1000));
  return members;
}

/** Re-admit every same-run input, reacquire the local Windows host SDK, and run the original source-JIT
 * isolator, static SDK, adapter and component producers. Mutates process.env for the in-process host-SDK
 * producer and the child toolchain PATH, then exports the component root through GITHUB_ENV. */
function buildComponent() {
  const env = process.env, {deadline, pins, hostSdk} = jobEnvironment(env);
  Object.assign(env, hostSdk);
  const suffix = ciSuffix(env), paths = ciSources.ciPaths(), temp = paths.runnerTemp,
    incoming = path.join(temp, "incoming"),
    rootsFile = path.join(temp, `windows-dpcpp-static-source-roots-${suffix}.json`),
    descriptorFile = path.join(temp, `windows-dpcpp-static-descriptor-${suffix}.json`),
    toolsManifest = path.join(temp, `windows-dpcpp-static-jit-tools-${suffix}.json`),
    componentRoot = path.join("/tmp", `native-windows-dpcpp-static-${suffix}`),
    jitIsolation = path.join("/tmp", `mom-native-windows-dpcpp-jit-isolated-${suffix}`),
    sdkRoot = path.join("/tmp", `mom-native-windows-dpcpp-sdk-${suffix}`),
    adaptersRoot = path.join("/tmp", `mom-native-windows-dpcpp-adapters-${suffix}`);
  const source = ciSources.linuxSourceStatePaths(paths, env), dpcpp = ciSources.dpcppCudaInputsStatePaths(paths, env),
    hermes = ciSources.hermesStatePaths(paths, env), jitName = `mom-native-windows-jit-${suffix}`,
    toolsName = `native-linux-llvm-tools-${suffix}`, ioName = `native-windows-controller-io-${suffix}`,
    windowsHermesName = `native-windows-controller-hermes-${suffix}`,
    sourceTar = path.join(incoming, path.basename(source.sourceTar)),
    dpcppTar = path.join(incoming, path.basename(dpcpp.sdkTar)),
    jitTar = path.join(incoming, `native-windows-jit-sdk-${suffix}-link.tar`),
    toolsTar = path.join(incoming, toolsName + ".tar"),
    ioTar = path.join(incoming, ioName + ".tar"),
    windowsHermesTar = path.join(incoming, windowsHermesName + ".tar"),
    linuxHermesTar = path.join(incoming, `native-linux-hermes-sdk-${suffix}.tar`),
    toolsParent = path.join(temp, `windows-dpcpp-static-tools-${suffix}`),
    windowsIoRoot = path.join(temp, ioName), windowsHermesRoot = path.join(temp, windowsHermesName),
    jitRoot = path.join("/tmp", jitName), jitSdk = path.join(jitRoot, "sdk");
  for (const output of [rootsFile, descriptorFile, toolsManifest, componentRoot, toolsParent, jitIsolation, sdkRoot, adaptersRoot]) {
    if (fs.existsSync(output)) {throw new Error("Windows DPC++ CI refuses an existing CI work-root output: " + path.basename(output));}
  }

  restore(deadline, sourceTar, pins.source, [source.sourceName], "/tmp", 8 * 1024 ** 3);
  ciSources.verifyLinuxSources(paths, source, fs.realpathSync("/usr/bin/git"));
  restore(deadline, dpcppTar, pins.dpcpp, [dpcpp.name], "/tmp", 8 * 1024 ** 3);
  dpcppInputs.auditDpcppCudaInputs(dpcpp, windowsGpuDpcppManifest(dpcpp.manifest, pins.dpcppManifest, pins.source), pins.dpcppManifest);
  restore(deadline, jitTar, pins.jit, [jitName], "/tmp", 12 * 1024 ** 3);
  fs.mkdirSync(toolsParent, {mode: 0o700});
  restore(deadline, toolsTar, pins.tools, [toolsName], toolsParent, 2 * 1024 ** 3);
  restore(deadline, ioTar, pins.io, [ioName], temp, 2 * 1024 ** 3);
  restore(deadline, windowsHermesTar, pins.hermes, [windowsHermesName], temp, 2 * 1024 ** 3);
  restore(deadline, linuxHermesTar, pins.linuxHermes, [hermes.sourceName, hermes.sdkName], "/tmp", 12 * 1024 ** 3);

  const tools = linuxLlvmToolHandoffInputs(path.join(toolsParent, toolsName), {handoffSha256: pins.toolsHandoff}),
    io = windowsControllerHandoffInputs(windowsIoRoot, {kind: "io", handoffSha256: pins.ioHandoff}),
    windowsHermes = windowsControllerHandoffInputs(windowsHermesRoot, {kind: "hermes", handoffSha256: pins.hermesHandoff}),
    linuxHermes = nativeHermesInputs(path.join(hermes.sdkRoot, "result.json"));
  if (io.kind !== "io" || windowsHermes.kind !== "hermes" || !windowsHermes.intelRuntime) {
    throw new Error("Windows DPC++ composition requires admitted same-run I/O and Hermes handoffs");
  }
  const roots = /** @type {Record<string, string> | undefined} */ (readBuildJson(source.rootsFile)),
    compilerNames = Object.keys(sdkComponents("compiler")),
    windowsNames = Object.keys(sdkComponents("windows")), preparedWindows = path.join(source.prepared, "windows");
  if (!roots || !hasExactBuildFields(roots, compilerNames) ||
      windowsNames.some(name => roots[name] !== path.join(source.source, "roots", name)) || fs.existsSync(preparedWindows)) {
    throw new Error("Pinned Windows source roots are incomplete or the preparation output is not fresh");
  }
  const windowsRoots = Object.fromEntries(windowsNames.map(name => [name, roots[name]]));
  fs.writeFileSync(rootsFile, JSON.stringify(windowsRoots) + "\n", {flag: "wx", mode: 0o444});
  runNode(deadline, "scripts/prepare-native-sdk-sources.js", ["windows", rootsFile, preparedWindows], 10 * 60 * 1000);
  const sourcesFile = path.join(preparedWindows, "sources.json"), hostState = ciWindowsLlvm.windowsHostSdkStatePaths(),
    hostProducer = ciWindowsLlvm.buildWindowsHostSdkSourcesCi(), hostPins = windowsGpuHostSdkPins(hostState.sdkOutput, hostProducer),
    clangCl = path.join(intelRoot, "bin/icx-cl"),
    llvmDescriptor = ciWindowsLlvm.composeWindowsLlvmSdkDescriptor({windowsSdkRoot: hostState.sdkOutput, windowsSdkPins: hostPins,
      hermesRoot: windowsHermesRoot, hermesExpectation: {kind: "hermes", handoffSha256: pins.hermesHandoff},
      tools: {clangCl: {path: clangCl, sha256: fileHash(clangCl)},
        lldLink: {path: tools.lldLink, sha256: fileHash(tools.lldLink), linkTarget: "lld"},
        llvmNm: {path: tools.llvmNm, sha256: fileHash(tools.llvmNm)},
        clangResourceInclude: path.join(intelRoot, "lib/clang/23/include")}}),
    config = composeWindowsApplicationConfig({llvmDescriptor: llvmDescriptor.descriptor, io, hermes: windowsHermes,
      hermesAot: linuxHermes, tools: {shermaot: linuxHermes.artifacts["compiler"], llvmReadobj: path.join(tools.toolRoot, "llvm-readobj")}, allowedImports});
  fs.writeFileSync(descriptorFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o444});
  fs.writeFileSync(toolsManifest, JSON.stringify({version: 1, tools: {llvmAr: path.join(tools.toolRoot, "llvm-ar"),
    llvmNm: tools.llvmNm, llvmObjcopy: path.join(tools.toolRoot, "llvm-objcopy")}}, null, 2) + "\n", {flag: "wx", mode: 0o444});

  const sourceJitResultSha256 = fileHash(path.join(jitSdk, "result.json"));
  env["PATH"] = [path.join(dpcpp.dpcppRoot, "bin"), path.join(intelRoot, "bin"), tools.toolRoot, "/usr/local/bin", "/usr/bin", "/bin"].join(":");
  env["LD_LIBRARY_PATH"] = [path.join(dpcpp.dpcppRoot, "lib"), path.join(intelRoot, "lib"), "/usr/lib/x86_64-linux-gnu"].join(":");
  runNode(deadline, "scripts/isolate-native-windows-jit.js", [jitSdk, sourceJitResultSha256, toolsManifest, jitIsolation], 15 * 60 * 1000);
  env["MOM_AOT_DPCPP_JIT_RECEIPT"] = path.join(jitIsolation, "result.json");
  runNode(deadline, "scripts/build-native-windows-sdk.js", [sourcesFile, descriptorFile, dpcpp.dpcppRoot, dpcpp.cudaRoot, sdkRoot], 30 * 60 * 1000);
  runNode(deadline, "scripts/prepare-native-windows-adapters.js",
    [path.join(sdkRoot, "result.json"), descriptorFile, dpcpp.cudaRoot, adaptersRoot], 10 * 60 * 1000);
  runNode(deadline, "scripts/native-windows-dpcpp-component.js", [path.join(adaptersRoot, "result.json"), dpcpp.cudaRoot, componentRoot], 5 * 60 * 1000);
  windowsDpcppComponentInputs(componentRoot);
  for (const file of [sourceTar, dpcppTar, jitTar, toolsTar, ioTar, windowsHermesTar, linuxHermesTar]) {fs.unlinkSync(file);}
  fs.appendFileSync(requiredEnv(env, "GITHUB_ENV"), [
    `MOM_NATIVE_WINDOWS_DPCPP_STATIC_DESCRIPTOR_FILE=${descriptorFile}`,
    `MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT=${componentRoot}`,
  ].join("\n") + "\n", "utf8");
}

/** Revalidate the fresh same-run component, pack exactly its manifest-listed members into
 * RUNNER_TEMP/native-windows-dpcpp-static-<run>-<attempt>.tar, and write the hash-bound job outputs.
 * @param {NodeJS.ProcessEnv} env */
function sealComponent(env) {
  const suffix = ciSuffix(env),
    root = path.join("/tmp", `native-windows-dpcpp-static-${suffix}`),
    archive = path.join(fs.realpathSync(requiredEnv(env, "RUNNER_TEMP")), `native-windows-dpcpp-static-${suffix}.tar`),
    receipt = path.join(root, "component.json"), result = windowsDpcppComponentInputs(root),
    {value: metadata, sha256: receiptSha256} = readBuildJsonSnapshot(receipt);
  if (env["MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT"] !== root || fs.existsSync(archive) ||
      !result.files[receipt] || result.files[receipt] !== receiptSha256) {
    throw new Error("Windows DPC++ component does not match the fresh same-run seal paths");
  }
  const name = path.basename(root), rows = [...metadata.link.libraries, ...metadata.link.wholeArchives, ...metadata.link.objects],
    expected = [name, name + "/component.json", name + "/lib", name + "/objects", ...rows.map(row => name + "/" + row)].sort(),
    packed = ciTar.packState(archive, path.dirname(root), [name], 5 * 60 * 1000),
    members = ciTar.listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort();
  if (!packed.bytes || !Number.isSafeInteger(packed.bytes) || members.join("\0") !== expected.join("\0") ||
      ciTar.tarExpandedBytes(archive, members.length) > 1024 ** 3 || fileHash(receipt) !== result.files[receipt]) {
    throw new Error("Windows DPC++ component bundle is incomplete or changed while packing");
  }
  fs.appendFileSync(requiredEnv(env, "GITHUB_OUTPUT"), `bundle_sha256=${fileHash(archive)}\ncomponent_manifest_sha256=${fileHash(receipt)}\n` +
    `artifact_name=native-windows-dpcpp-static-${suffix}.tar\n`, "utf8");
}

/** Dispatch one workflow step by name. @param {string | undefined} command */
function main(command) {
  if (command === "preflight") {
    jobEnvironment(process.env);
  } else if (command === "build") {
    buildComponent();
  } else if (command === "seal") {
    sealComponent(process.env);
  } else {
    throw new Error("Usage: node scripts/ci/windows-dpcpp.js preflight|build|seal");
  }
}

if (require.main === module) {
  try {
    main(process.argv[2]);
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}

module.exports = {jobEnvironment, readPins, pinArchive, buildComponent, sealComponent};
