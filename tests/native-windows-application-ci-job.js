"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto"),
  {spawnSync} = require("node:child_process"), {test} = require("node:test");
const job = require("../scripts/ci/windows-application"), {fileHash, readBuildJsonSnapshot} = require("../scripts/build-source-inputs");

const repo = path.resolve(__dirname, ".."), profiles = job.profiles,
  pinNames = ["CORE_BUNDLE", "CORE_HANDOFF", "DPCPP_COMPONENT_BUNDLE", "DPCPP_COMPONENT_MANIFEST", "HOST_COMPONENT_BUNDLE",
    "HOST_COMPONENT_MANIFEST", "IO_BUNDLE", "IO_HANDOFF", "HERMES_BUNDLE", "HERMES_HANDOFF", "LINUX_HERMES_BUNDLE",
    "LINUX_TOOLS_BUNDLE", "LINUX_TOOLS_HANDOFF", "DPCPP_INPUTS_BUNDLE", "DPCPP_INPUTS_MANIFEST", "LINUX_SOURCES_BUNDLE",
    "HIP_HEADERS_BUNDLE", "HIP_HEADERS_HANDOFF", "CUDA_IMAGES_BUNDLE"],
  ordinaryWindowsImports = ["advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "crypt32.dll", "dbghelp.dll", "dnsapi.dll", "gdi32.dll",
    "iphlpapi.dll", "kernel32.dll", "ncrypt.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "powrprof.dll", "psapi.dll",
    "rpcrt4.dll", "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "user32.dll", "userenv.dll", "uuid.dll",
    "version.dll", "winhttp.dll", "winmm.dll", "wsock32.dll", "ws2_32.dll"];

/** @param {import("node:test").TestContext} t */
function runnerTemp(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync("/var/tmp"), "mom-windows-application-job-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

/** @param {number} seconds */
function deadlineIn(seconds) {
  return String(Math.floor(Date.now() / 1000) + seconds);
}

/** @param {string} value */
function digest(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

/** Compose an actual descriptor using the public composer and reader-result-shaped fixture. @param {string} suffix @param {string[]} allowedImports @returns {any} */
function composeFixture(suffix, allowedImports = ordinaryWindowsImports) {
  const {composeWindowsApplicationConfig} = require("../scripts/native-windows-application-config"),
    sha256 = "a".repeat(64), at = (/** @type {string} */ root, /** @type {string} */ file) => path.join(root, file),
    ioRoot = `/tmp/native-windows-controller-io-${suffix}`, hermesRoot = `/tmp/native-windows-controller-hermes-${suffix}`,
    sdkRoot = `/tmp/windows-host-sdk-${suffix}`, intelRoot = at(hermesRoot, "intel-runtime/lib"),
    intelLibraries = Object.fromEntries(["libircmt.lib", "libirc.lib", "libmmt.lib", "svml_dispmt.lib", "libdecimal.lib"]
      .map(name => [name, at(intelRoot, name)])), hermesCompiler = `/tmp/linux-hermes-${suffix}/build/bin/shermes`,
    gpuHandoffs = profiles.map((profile, index) => ({profile, receiptSha256: String(index + 1).repeat(64),
      configSha256: String(index + 2).repeat(64), handoffSha256: String(index + 3).repeat(64)})),
    application = {core: "/app/core.lib", gpuReceipts: profiles.map(profile => `/app/${profile}/output/result.json`), gpuHandoffs,
      defaultCompiler: "dpcpp", tools: {llvmAr: "/tools/llvm-ar.exe", llvmObjcopy: "/tools/llvm-objcopy.exe"},
      sdk: {libraries: [], wholeArchives: [], objects: [], includeRoots: ["/cuda/include", "/hip/include"],
        nvmlHeader: "/cuda/include/nvml.h", hipDriverLibrary: "/host/runtime/amdhip64.lib",
        dpcppComponent: "/components/dpcpp", hostComponent: "/components/host", cudaImages: "/images/cuda-images.inc"}},
    input = {
      llvmDescriptor: {version: 1, tools: {clangCl: "/tools/icx-cl.exe", lldLink: "/tools/lld-link.exe", llvmNm: "/tools/llvm-nm.exe",
        clangResourceInclude: "/tools/resource/include"}, windowsSdk: {sourceRoot: sdkRoot, overlay: at(sdkRoot, "overlay.json"),
        includeRoots: [at(sdkRoot, "msvc/include"), at(sdkRoot, "sdk/include")],
        libraryRoots: [at(sdkRoot, "sdk/lib/ucrt"), at(sdkRoot, "sdk/lib/um"), at(sdkRoot, "msvc/lib"), intelRoot],
        msvcLibraries: [at(sdkRoot, "msvc/lib/libcmt.lib"), ...Object.values(intelLibraries)],
        ucrtLibraries: [at(sdkRoot, "sdk/lib/ucrt/libvcruntime.lib")], importLibraries: [at(sdkRoot, "sdk/lib/um/kernel32.lib")]}},
      io: {kind: "io", handoff: at(ioRoot, "handoff.json"), handoffSha256: sha256, files: {},
        libraries: {sourceInclude: at(ioRoot, "openssl-sdk/include"), ssl: at(ioRoot, "openssl-sdk/lib/libssl.lib"),
          crypto: at(ioRoot, "openssl-sdk/lib/libcrypto.lib"), libuv: at(ioRoot, "io-sdk/lib/uv_a.lib"),
          zlib: at(ioRoot, "io-sdk/lib/zlibstatic.lib"), libuvInclude: at(ioRoot, "io-sdk/include"),
          zlibInclude: at(ioRoot, "io-sdk/include"), zlibGeneratedInclude: at(ioRoot, "io-sdk/include")}},
      hermes: {kind: "hermes", handoff: at(hermesRoot, "handoff.json"), handoffSha256: sha256, files: {},
        hermes: {source: at(hermesRoot, "hermes/source"), build: at(hermesRoot, "hermes/build")},
        intelRuntime: {libRoot: intelRoot, libraries: intelLibraries,
          licenses: {license: at(hermesRoot, "intel-runtime/licenses/LICENSE.rtf")}}},
      hermesAot: {artifacts: {compiler: hermesCompiler}, files: {[hermesCompiler]: sha256}},
      tools: {shermaot: hermesCompiler, llvmReadobj: "/tools/llvm-readobj.exe"},
      allowedImports, application};
  return composeWindowsApplicationConfig(input);
}

/** Same-run external bundle and receipt pins as the workflow passes them. @param {string} suffix @returns {NodeJS.ProcessEnv} */
function pinEnvironment(suffix) {
  const gpuPins = profiles.map((profile, index) => ({profile, bundleSha256: String(index + 1).repeat(64),
    receiptSha256: String(index + 2).repeat(64), configSha256: String(index + 3).repeat(64), handoffSha256: String(index + 4).repeat(64)}));
  return {...Object.fromEntries(pinNames.map(name => [`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`, "a".repeat(64)])),
    MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(gpuPins), MOM_NATIVE_CI_SUFFIX: suffix};
}

/** Recorded fakes for every reader the build admits restored inputs with; extraction creates the
 * minimal restored files those fakes and the build's own filesystem checks require.
 * @param {string} suffix @param {string} temp @param {{dpcppManifest:string,hostManifest:string}} manifests */
function fakeBuildDependencies(suffix, temp, manifests) {
  const restoreCalls = /** @type {Array<{file:string,destination:string,roots:string[]}>} */ ([]),
    observed = /** @type {any} */ ({compose: null, imageAudit: null, buildArgs: null, buildEnvironment: null}),
    hash = (/** @type {string} */ file) => file === "/opt/intel/oneapi/compiler/2026.2/bin/icx-cl" ? "f".repeat(64) : fileHash(file),
    put = (/** @type {string} */ file, value = "fixture\n") => {
      fs.mkdirSync(path.dirname(file), {recursive: true, mode: 0o700});
      fs.writeFileSync(file, value, {flag: "wx", mode: 0o444});
    },
    hermesCompiler = path.join(temp, "fixture-hermes", "shermes");
  put(hermesCompiler, "fixture hermes compiler\n");
  /** @type {Record<string, (root: string) => void>} */
  const restoredFiles = {
    [`mom-native-dpcpp-cuda-inputs-${suffix}`]: root => {put(path.join(root, "cuda/include/nvml.h")); put(path.join(root, "cuda/include/cuda.h"));},
    [`native-windows-dpcpp-static-${suffix}`]: root => put(path.join(root, "component.json"), manifests.dpcppManifest),
    [`native-windows-host-component-${suffix}`]: root => {
      put(path.join(root, "component.json"), manifests.hostManifest); put(path.join(root, "runtime/amdhip64.lib"));
    },
    [`native-rocm-hip-headers-${suffix}`]: root => put(path.join(root, "include/hip/hip_runtime_api.h")),
    [`mom-native-cuda-images-handoff-${suffix}`]: root => put(path.join(root, "cuda-images.inc")),
    [`native-linux-llvm-tools-${suffix}`]: root => {
      for (const name of ["lld-link", "llvm-nm", "llvm-ar", "llvm-objcopy", "llvm-readobj"]) {put(path.join(root, "bin", name));}
    },
  };
  const ci = {
    ciPaths: () => ({suffix, runnerTemp: temp}),
    linuxSourceStatePaths: () => ({sourceName: `mom-native-linux-source-prerequisites-${suffix}`,
      sourceTar: path.join(temp, `native-linux-sdk-state-${suffix}-sources.tar`)}),
    dpcppCudaInputsStatePaths: () => ({name: `mom-native-dpcpp-cuda-inputs-${suffix}`,
      sdkTar: path.join(temp, `native-linux-dpcpp-cuda-inputs-${suffix}.tar`),
      manifest: path.join("/tmp", `mom-native-dpcpp-cuda-inputs-${suffix}`, "inputs.json"),
      cudaRoot: path.join("/tmp", `mom-native-dpcpp-cuda-inputs-${suffix}`, "cuda")}),
    hermesStatePaths: () => ({sourceName: `mom-native-hermes-prerequisites-${suffix}`, sdkName: `mom-native-hermes-sdk-${suffix}`,
      sdkRoot: path.join("/tmp", `mom-native-hermes-sdk-${suffix}`)}),
    cudaImagesStatePaths: () => ({handoffRoot: path.join("/tmp", `mom-native-cuda-images-handoff-${suffix}`)}),
    listState: (/** @type {string} */ _file, /** @type {string[]} */ roots) => roots,
    tarExpandedBytes: () => 1024,
    extractState: (/** @type {string} */ file, /** @type {string} */ destination, /** @type {string[]} */ roots) => {
      restoreCalls.push({file, destination, roots});
      for (const root of roots) {
        fs.mkdirSync(path.join(destination, root), {mode: 0o700});
        restoredFiles[root]?.(path.join(destination, root));
      }
    },
    verifyLinuxSources: () => undefined,
    auditDpcppCudaInputs: () => ({}),
    auditLinuxNativeSdkDescriptorCudaImages: (/** @type {any} */ state) => {
      observed.imageAudit = Object.keys(state).sort();
      return {include: path.join(state.images.handoffRoot, "cuda-images.inc"), includeSha256: "e".repeat(64)};
    },
    buildWindowsHostSdkSourcesCi: () => ({sdk_receipt_sha256: "a".repeat(64), descriptor_sha256: "b".repeat(64),
      files_provenance_sha256: "c".repeat(64)}),
    composeWindowsLlvmSdkDescriptor: () => ({descriptor: {version: 1, tools: {}, windowsSdk: {}}}),
  };
  const spawn = (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {any} */ options) => {
    observed.buildArgs = {command, args};
    observed.buildEnvironment = options.env;
    const index = args.indexOf("--native"), config = args[index + 1] || "", output = args[index + 2] || "", binary = path.join(output, "mom.exe");
    fs.mkdirSync(output, {mode: 0o700}); put(binary, "MZ\n");
    const receipt = {target: "x86_64-pc-windows-msvc", nativeRuntime: true, buildOnly: true, executed: false, publicationQualified: false,
      configHash: hash(config), artifact: {path: binary, bytes: fs.statSync(binary).size, sha256: hash(binary)},
      application: {graphs: profiles.map(profile => ({factory: "mom_native_gpu_" + profile.replaceAll("-", "_")}))}};
    put(path.join(output, "result.json"), JSON.stringify(receipt) + "\n"); put(path.join(output, "entrypoint.json"), "{}\n");
    return {status: 0, signal: null, error: null};
  };
  const deps = {ci, sourceInputs: {fileHash: hash, readBuildJsonSnapshot},
    linuxLlvmToolHandoffInputs: (/** @type {string} */ root) => ({toolRoot: path.join(root, "bin"), lldLink: path.join(root, "bin/lld-link"),
      llvmNm: path.join(root, "bin/llvm-nm")}),
    windowsControllerHandoffInputs: (/** @type {string} */ root, /** @type {any} */ expected) => ({kind: expected.kind,
      handoff: path.join(root, "handoff.json"), handoffSha256: expected.handoffSha256, files: {},
      ...(expected.kind === "hermes" ? {intelRuntime: {}} : {libraries: {}})}),
    windowsNativeCoreHandoffInputs: (/** @type {string} */ root) => ({archive: path.join(root, "cpu-core.lib")}),
    windowsDpcppComponentInputs: (/** @type {string} */ root) => ({files: {[path.join(root, "component.json")]: hash(path.join(root, "component.json"))}}),
    windowsHostComponentInputs: (/** @type {string} */ root) => ({files: {[path.join(root, "component.json")]: hash(path.join(root, "component.json"))},
      runtimeInputs: {amdhip64: path.join(root, "runtime/amdhip64.lib")}}),
    windowsGpuHandoffInputs: (/** @type {string} */ _file, /** @type {any} */ expected) => ({profile: expected.profile}),
    rocmHipHeaderInputs: (/** @type {string} */ root) => ({include: path.join(root, "include"), files: {}}),
    nativeHermesInputs: () => ({artifacts: {compiler: hermesCompiler}, files: {[hermesCompiler]: hash(hermesCompiler)}}),
    composeWindowsApplicationConfig: (/** @type {any} */ input) => {observed.compose = input; return {application: input.application};},
    windowsSystemImports: () => ordinaryWindowsImports,
    windowsGpu: {
      windowsGpuHostSdkPins: () => ({resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64), filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)}),
      windowsGpuDpcppManifest: (/** @type {string} */ _file, /** @type {string} */ manifestSha256, /** @type {string} */ sourceSha256) =>
        ({source: {tarSha256: sourceSha256}, manifestSha256}),
    },
    spawnSync: spawn};
  return {deps: /** @type {any} */ (deps), restoreCalls, observed};
}

test.describe("Windows full-application CI job", {skip: process.platform !== "linux" && "Linux CI job harness"}, () => {
  test("full application admits the maintained ordinary Windows DLL set but no application DLLs", () => {
    const selected = require("../scripts/build-native-windows-controller").windowsSystemImports();
    assert.deepEqual(selected, ordinaryWindowsImports);
    const config = composeFixture(`ci-${Date.now()}${process.pid}-1`, selected);
    assert.deepEqual(config.windowsSdk.allowedImports, ordinaryWindowsImports);
    assert.throws(() => composeFixture(`${Date.now()}${process.pid}-2`, [...selected, "mom.exe.dll"]), /ordinary system DLLs/);
  });

  test("build restores the planned flat cohort and invokes the public release command", t => {
    const temp = runnerTemp(t), runId = `${Date.now()}${process.pid}`, suffix = `ci-${runId}-1`, deadline = deadlineIn(900),
      coreName = `native-windows-core-${suffix}`, dpcppComponentName = `native-windows-dpcpp-static-${suffix}`,
      hostComponentName = `native-windows-host-component-${suffix}`, ioName = `native-windows-controller-io-${suffix}`,
      windowsHermesName = `native-windows-controller-hermes-${suffix}`, linuxToolsName = `native-linux-llvm-tools-${suffix}`,
      dpcppInputsName = `mom-native-dpcpp-cuda-inputs-${suffix}`, hipHeadersName = `native-rocm-hip-headers-${suffix}`,
      tempRoots = [ioName, windowsHermesName, linuxToolsName],
      restoreRoots = [coreName, dpcppComponentName, hostComponentName, ioName, windowsHermesName,
        `mom-native-hermes-prerequisites-${suffix}`, `mom-native-hermes-sdk-${suffix}`, linuxToolsName, dpcppInputsName,
        `mom-native-linux-source-prerequisites-${suffix}`, hipHeadersName, `mom-native-cuda-images-handoff-${suffix}`,
        ...profiles.map(profile => `mom-native-windows-gpu-${profile}-${suffix}`)];
    for (const root of restoreRoots) {
      const location = path.join(tempRoots.includes(root) ? temp : "/tmp", root);
      assert.equal(fs.existsSync(location), false, "fixture restore root is new: " + root);
      t.after(() => fs.rmSync(location, {recursive: true, force: true}));
    }
    const manifests = {dpcppManifest: JSON.stringify({fixture: "dpcpp"}) + "\n", hostManifest: JSON.stringify({fixture: "host"}) + "\n"},
      bundleNames = {CORE_BUNDLE: `${coreName}.tar`, DPCPP_COMPONENT_BUNDLE: `${dpcppComponentName}.tar`,
        HOST_COMPONENT_BUNDLE: `${hostComponentName}.tar`, IO_BUNDLE: `${ioName}.tar`, HERMES_BUNDLE: `${windowsHermesName}.tar`,
        LINUX_HERMES_BUNDLE: `native-linux-hermes-sdk-${suffix}.tar`, LINUX_TOOLS_BUNDLE: `${linuxToolsName}.tar`,
        DPCPP_INPUTS_BUNDLE: `native-linux-dpcpp-cuda-inputs-${suffix}.tar`, LINUX_SOURCES_BUNDLE: `native-linux-sdk-state-${suffix}-sources.tar`,
        HIP_HEADERS_BUNDLE: `native-rocm-hip-headers-${suffix}.tar`, CUDA_IMAGES_BUNDLE: `native-linux-cuda-images-${suffix}.tar`},
      incoming = path.join(temp, "incoming");
    fs.mkdirSync(incoming, {mode: 0o700});
    /** @param {string} name */
    const download = name => {
      const file = path.join(incoming, name);
      fs.writeFileSync(file, `bounded fixture archive: ${name}\n`, {flag: "wx", mode: 0o444});
      return fileHash(file);
    };
    const pins = Object.fromEntries(pinNames.map(name => [`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`, "a".repeat(64)]));
    for (const [name, file] of Object.entries(bundleNames)) {pins[`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`] = download(file);}
    pins["MOM_NATIVE_WINDOWS_APPLICATION_DPCPP_COMPONENT_MANIFEST_SHA256"] = digest(manifests.dpcppManifest);
    pins["MOM_NATIVE_WINDOWS_APPLICATION_HOST_COMPONENT_MANIFEST_SHA256"] = digest(manifests.hostManifest);
    pins["MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON"] = JSON.stringify(profiles.map(profile => ({profile,
      bundleSha256: download(`native-windows-gpu-${profile}-${suffix}.tar`), receiptSha256: "b".repeat(64),
      configSha256: "c".repeat(64), handoffSha256: "d".repeat(64)})));
    const output = path.join(temp, `native-windows-application-output-${suffix}`),
      configFile = path.join(temp, `native-windows-application-config-${suffix}.json`), githubOutput = path.join(temp, "github-output"),
      cudaInclude = path.join("/tmp", dpcppInputsName, "cuda/include"), hipInclude = path.join("/tmp", hipHeadersName, "include");
    fs.writeFileSync(githubOutput, "", {flag: "wx"});
    /** @type {NodeJS.ProcessEnv} */
    const environment = {PATH: process.env["PATH"], RUNNER_TEMP: temp, MOM_NATIVE_CI_SUFFIX: "ci-" + runId + "-1",
      GITHUB_OUTPUT: githubOutput, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "4", MOM_CI_DEADLINE_EPOCH: deadline, ...pins};
    const {deps, restoreCalls, observed} = fakeBuildDependencies(suffix, temp, manifests);
    job.runBuild(environment, deps);

    assert.equal(restoreCalls.length, 16);
    for (const row of restoreCalls) {
      assert.equal(path.dirname(row.file), incoming, "every download is read from the flat incoming directory");
      assert.equal(fs.existsSync(row.file), false, "verified transport is discarded before native compilation");
      for (const root of row.roots) {assert.equal(fs.statSync(path.join(row.destination, root)).isDirectory(), true);}
    }
    assert.deepEqual(restoreCalls.filter(row => tempRoots.includes(row.roots[0] || "")).map(row => row.destination), Array(3).fill(temp));
    assert.ok(restoreCalls.filter(row => !tempRoots.includes(row.roots[0] || "")).every(row => row.destination === "/tmp"));
    const application = observed.compose.application;
    assert.deepEqual(application.gpuHandoffs.map((/** @type {{profile:string}} */ row) => row.profile).sort(), [...profiles].sort());
    assert.equal(application.core, path.join("/tmp", coreName, "cpu-core.lib"));
    assert.deepEqual(application.sdk.includeRoots, [cudaInclude, hipInclude]);
    assert.equal(application.sdk.nvmlHeader, path.join(cudaInclude, "nvml.h"));
    assert.equal(application.sdk.hipDriverLibrary, path.join("/tmp", hostComponentName, "runtime/amdhip64.lib"));
    assert.deepEqual(observed.compose.allowedImports, ordinaryWindowsImports);
    assert.deepEqual(observed.imageAudit, ["images"]);
    assert.equal(environment["MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH"], deadline, "host-SDK acquisition shares the aggregate deadline");
    assert.equal(environment["MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR"],
      path.join(temp, `mom-native-windows-host-sdk-${suffix}`, `native-windows-host-sdk-${suffix}`));
    assert.equal(environment["MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR"], path.join(temp, `native-windows-host-sdk-${suffix}-metadata.tar`));
    const buildEnvironment = observed.buildEnvironment, buildArgs = observed.buildArgs;
    assert.deepEqual(Object.keys(buildEnvironment).filter(key => key.startsWith("MOM_NATIVE_WINDOWS_APPLICATION_") || key === "MOM_CI_DEADLINE_EPOCH"),
      [], "admitted transport pins and changing deadlines must not invalidate compiler caches");
    assert.equal(buildEnvironment.MOM_AOT_BUILD_ONLY, "1");
    assert.equal(buildEnvironment.MOM_BUILD_JOBS, "4");
    assert.equal(buildArgs.command, path.join(repo, "scripts/run-bounded-native-ci.sh"), "the final link runs inside the host memory scope");
    assert.equal(buildArgs.args[0], fs.realpathSync("/usr/bin/timeout"), "the aggregate child deadline remains inside that scope");
    assert.deepEqual(buildArgs.args.slice(buildArgs.args.indexOf("--native") - 4), ["npm", "run", "build:release", "--", "--native", configFile, output]);
    assert.equal(fs.existsSync(path.join(output, "mom.exe")), true);
    const values = Object.fromEntries(fs.readFileSync(githubOutput, "utf8").trim().split("\n").map(line => line.split("=")));
    assert.deepEqual(values, {build_output: output, config_file: configFile, executable: path.join(output, "mom.exe"),
      result_file: path.join(output, "result.json"), entrypoint_file: path.join(output, "entrypoint.json")});
  });

  test("build refuses an unsupported job envelope before touching any download", t => {
    const temp = runnerTemp(t), suffix = `ci-${Date.now()}${process.pid}-1`,
      deps = /** @type {any} */ ({ci: {ciPaths: () => ({suffix, runnerTemp: temp})}}),
      environment = {...pinEnvironment(suffix), RUNNER_TEMP: temp, MOM_AOT_BUILD_ONLY: "1", MOM_CI_DEADLINE_EPOCH: deadlineIn(900)};
    assert.throws(() => job.runBuild({...environment, MOM_BUILD_JOBS: "3"}, deps), /MOM_BUILD_JOBS/);
    assert.throws(() => job.runBuild({...environment, MOM_BUILD_JOBS: "4", MOM_CI_DEADLINE_EPOCH: "1"}, deps),
      /aggregate deadline is invalid or expired/);
  });

  test("build rejects a download whose bytes differ from its external pin before tar reads it", t => {
    const temp = runnerTemp(t), suffix = `ci-${Date.now()}${process.pid}-1`, download = path.join(temp, "incoming", `native-windows-core-${suffix}.tar`),
      {deps, restoreCalls} = fakeBuildDependencies(suffix, temp, {dpcppManifest: "{}\n", hostManifest: "{}\n"});
    fs.mkdirSync(path.dirname(download), {mode: 0o700});
    fs.writeFileSync(download, "tampered\n", {flag: "wx", mode: 0o444});
    assert.throws(() => job.runBuild({...pinEnvironment(suffix), RUNNER_TEMP: temp, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "4",
      MOM_CI_DEADLINE_EPOCH: deadlineIn(900)}, deps), /differs from its external same-run SHA-256: native-windows-core-/);
    assert.deepEqual(restoreCalls, []);
    assert.equal(fs.existsSync(download), true, "a rejected transport is left for diagnosis");
  });

  test("pins accepts exactly five complete same-run profiles", t => {
    const temp = runnerTemp(t), output = path.join(temp, "github-output");
    fs.writeFileSync(output, "", {flag: "wx"});
    const suffix = `ci-${Date.now()}-1`, base = /** @type {NodeJS.ProcessEnv} */ ({...pinEnvironment(suffix), GITHUB_OUTPUT: output});
    job.runPins(base);
    const duplicated = JSON.parse(String(base["MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON"]));
    duplicated[4].profile = duplicated[0].profile;
    assert.throws(() => job.runPins({...base, MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(duplicated)}), /exact five/);
    const extraKey = JSON.parse(String(base["MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON"]));
    extraKey[1].note = "x";
    assert.throws(() => job.runPins({...base, MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(extraKey)}), /exact five/);
    assert.throws(() => job.runPins({...base, MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: "["}), /invalid JSON/);
    assert.throws(() => job.runPins({...base, MOM_NATIVE_WINDOWS_APPLICATION_CORE_BUNDLE_SHA256: "bad"}), /malformed/);
    assert.throws(() => job.runPins({...base, MOM_NATIVE_CI_SUFFIX: "0/1"}), /MOM_NATIVE_CI_SUFFIX/);
    assert.equal(fs.readFileSync(output, "utf8"), "");
  });

  test("seal refuses missing or expired aggregate deadlines", t => {
    const temp = runnerTemp(t), runId = `${Date.now()}${process.pid}`, suffix = `ci-${runId}-1`,
      buildRoot = path.join(temp, `native-windows-application-output-${suffix}`),
      evidenceRoot = path.join("/tmp", `native-windows-application-build-${suffix}`),
      archive = path.join(temp, `native-windows-application-build-${suffix}.tar`),
      base = {RUNNER_TEMP: temp, MOM_NATIVE_CI_SUFFIX: "ci-" + runId + "-1", MOM_CI_DEADLINE_EPOCH: deadlineIn(300),
        BUILD_OUTPUT: buildRoot, CONFIG_FILE: path.join(temp, `native-windows-application-config-${suffix}.json`),
        EXECUTABLE: path.join(buildRoot, "mom.exe"), RESULT_FILE: path.join(buildRoot, "result.json"),
        ENTRYPOINT_FILE: path.join(buildRoot, "entrypoint.json")};
    const {MOM_CI_DEADLINE_EPOCH: _unused, ...missing} = base;
    assert.throws(() => job.runSeal(missing), /aggregate deadline is invalid or expired/);
    assert.throws(() => job.runSeal({...base, MOM_CI_DEADLINE_EPOCH: "1"}), /aggregate deadline is invalid or expired/);
    const cli = spawnSync(process.execPath, [path.join(repo, "scripts/ci/windows-application.js"), "seal"],
      {env: {...process.env, ...base, MOM_CI_DEADLINE_EPOCH: "1"}, encoding: "utf8", timeout: 30000});
    assert.equal(cli.status, 1);
    assert.match(cli.stderr, /^\[native-ci\] .*aggregate deadline is invalid or expired/m);
    assert.equal(fs.existsSync(evidenceRoot), false); assert.equal(fs.existsSync(archive), false);
  });

  test("the public config composer and seal produce only the raw executable plus compact evidence", t => {
    const ci = require("../scripts/native-ci-tar"), temp = runnerTemp(t),
      suffix = `ci-${Date.now()}${process.pid}-1`, config = composeFixture(suffix), buildRoot = path.join(temp, `native-windows-application-output-${suffix}`),
      configFile = path.join(temp, `native-windows-application-config-${suffix}.json`), executable = path.join(buildRoot, "mom.exe"),
      resultFile = path.join(buildRoot, "result.json"), entrypointFile = path.join(buildRoot, "entrypoint.json"),
      evidenceName = `native-windows-application-build-${suffix}`, evidenceRoot = path.join("/tmp", evidenceName),
      archive = path.join(temp, evidenceName + ".tar"), output = path.join(temp, "github-output");
    assert.equal(fs.existsSync(buildRoot), false); assert.equal(fs.existsSync(evidenceRoot), false);
    t.after(() => fs.rmSync(evidenceRoot, {recursive: true, force: true}));
    fs.mkdirSync(buildRoot, {mode: 0o700});
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o444});
    fs.writeFileSync(executable, Buffer.from("MZ\0fixture-only\n"), {flag: "wx", mode: 0o555});
    const executableSha256 = fileHash(executable), configSha256 = fileHash(configFile), executableBytes = fs.statSync(executable).size,
      result = {target: "x86_64-pc-windows-msvc", nativeRuntime: true, buildOnly: true, executed: false, publicationQualified: false,
        configHash: configSha256, artifact: {path: executable, bytes: executableBytes, sha256: executableSha256},
        application: {graphs: profiles.map(profile => ({factory: "mom_native_gpu_" + profile.replaceAll("-", "_")}))}};
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2) + "\n", {flag: "wx", mode: 0o444});
    const entrypoint = {version: 1, target: "win32", configFile, configHash: configSha256,
      source: {path: executable, sha256: executableSha256}, artifact: result.artifact, producerReceipt: {path: resultFile, sha256: fileHash(resultFile)},
      buildOnly: true, publicationQualified: false};
    fs.writeFileSync(entrypointFile, JSON.stringify(entrypoint, null, 2) + "\n", {flag: "wx", mode: 0o444});
    fs.writeFileSync(output, "", {flag: "wx"});
    job.runSeal({RUNNER_TEMP: temp, MOM_NATIVE_CI_SUFFIX: suffix, GITHUB_OUTPUT: output,
      MOM_CI_DEADLINE_EPOCH: deadlineIn(300), BUILD_OUTPUT: buildRoot, CONFIG_FILE: configFile, EXECUTABLE: executable,
      RESULT_FILE: resultFile, ENTRYPOINT_FILE: entrypointFile});
    const values = Object.fromEntries(fs.readFileSync(output, "utf8").trim().split("\n").map(line => line.split("=")));
    assert.equal(values.executable_artifact_name, `native-windows-executable-${suffix}`);
    assert.equal(values.executable_sha256, executableSha256);
    assert.equal(Number(values.executable_bytes), executableBytes);
    assert.equal(values.build_handoff_artifact_name, evidenceName + ".tar");
    assert.equal(values.build_handoff_bundle_sha256, fileHash(archive));
    const members = ci.listState(archive, [evidenceName]).map(member => member.replace(/\/$/, "")).sort();
    assert.deepEqual(members, [evidenceName, ...["application-config.json", "entrypoint.json", "handoff.json", "result.json"]
      .map(name => `${evidenceName}/${name}`)].sort());
    const restored = path.join(temp, "restored-evidence"); fs.mkdirSync(restored, {mode: 0o700});
    ci.extractState(archive, restored, [evidenceName], [evidenceName], 30000);
    const handoffFile = path.join(restored, evidenceName, "handoff.json"),
      handoff = /** @type {{artifact:{sha256:string},gpuHandoffs:Array<{profile:string}>}} */ (
        JSON.parse(fs.readFileSync(handoffFile, "utf8")));
    assert.equal(values.build_handoff_sha256, fileHash(handoffFile));
    assert.equal(handoff.artifact.sha256, executableSha256);
    assert.deepEqual(handoff.gpuHandoffs.map(row => row.profile).sort(), [...profiles].sort());
  });
});
