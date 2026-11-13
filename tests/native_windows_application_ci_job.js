"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"),
  {spawnSync} = require("node:child_process"), {test} = require("node:test");

const repo = path.resolve(__dirname, ".."), workflowPath = path.join(repo, ".github/workflows/build-windows-application-job.yml"),
  workflow = fs.readFileSync(workflowPath, "utf8"), profiles = ["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"],
  pinNames = ["CORE_BUNDLE", "CORE_HANDOFF", "DPCPP_COMPONENT_BUNDLE", "DPCPP_COMPONENT_MANIFEST", "HOST_COMPONENT_BUNDLE",
    "HOST_COMPONENT_MANIFEST", "IO_BUNDLE", "IO_HANDOFF", "HERMES_BUNDLE", "HERMES_HANDOFF", "LINUX_HERMES_BUNDLE",
    "LINUX_TOOLS_BUNDLE", "LINUX_TOOLS_HANDOFF", "DPCPP_INPUTS_BUNDLE", "DPCPP_INPUTS_MANIFEST", "LINUX_SOURCES_BUNDLE",
    "HIP_HEADERS_BUNDLE", "HIP_HEADERS_HANDOFF", "CUDA_IMAGES_BUNDLE"],
  ordinaryWindowsImports = ["advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "crypt32.dll", "dbghelp.dll", "dnsapi.dll", "gdi32.dll",
    "iphlpapi.dll", "kernel32.dll", "ncrypt.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "powrprof.dll", "psapi.dll",
    "rpcrt4.dll", "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "user32.dll", "userenv.dll", "uuid.dll",
    "version.dll", "winhttp.dll", "winmm.dll", "wsock32.dll", "ws2_32.dll"];

/** @param {string} name */
function nodeBody(name) {
  const marker = "      - name: " + name, position = workflow.indexOf(marker), heredoc = "          node <<'NODE'\n",
    start = workflow.indexOf(heredoc, position) + heredoc.length, end = workflow.indexOf("\n          NODE", start);
  assert.ok(position >= 0 && start >= heredoc.length && end > start, "production Node heredoc exists: " + name);
  return workflow.slice(start, end).split("\n").map(line => line.slice(10)).join("\n");
}

/** @param {string} body @param {NodeJS.ProcessEnv} env */
function execNode(body, env) {
  return spawnSync(process.execPath, ["-e", body], {cwd: repo, env, encoding: "utf8", timeout: 30000, maxBuffer: 2 << 20});
}

/** @param {import("node:test").TestContext} t */
function runnerTemp(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync("/var/tmp"), "mom-windows-application-job-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

/** Compose an actual descriptor using the public composer and reader-result-shaped fixture. @param {string} suffix @param {string[]} allowedImports @returns {any} */
function composeFixture(suffix, allowedImports = ordinaryWindowsImports) {
  const {composeWindowsApplicationConfig} = require("../scripts/native-windows-application-config"),
    digest = "a".repeat(64), at = (/** @type {string} */ root, /** @type {string} */ file) => path.join(root, file),
    ioRoot = `/tmp/native-windows-controller-io-${suffix}`, hermesRoot = `/tmp/native-windows-controller-hermes-${suffix}`,
    sdkRoot = `/tmp/windows-host-sdk-${suffix}`, intelRoot = at(hermesRoot, "intel-runtime/lib"),
    intelLibraries = Object.fromEntries(["libircmt.lib", "libirc.lib", "libmmt.lib", "svml_dispmt.lib", "libdecimal.lib"]
      .map(name => [name, at(intelRoot, name)])), hermesCompiler = `/tmp/linux-hermes-${suffix}/build/bin/shermes`,
    gpuHandoffs = profiles.map((profile, index) => ({profile, runAttempt: suffix, receiptSha256: String(index + 1).repeat(64),
      configSha256: String(index + 2).repeat(64), handoffSha256: String(index + 3).repeat(64)})),
    application = {core: "/app/core.lib", gpuReceipts: profiles.map(profile => `/app/${profile}/output/result.json`), gpuHandoffs,
      defaultCompiler: "dpcpp", tools: {llvmAr: "/tools/llvm-ar.exe", llvmObjcopy: "/tools/llvm-objcopy.exe"},
      sdk: {libraries: [], wholeArchives: [], objects: [], includeRoots: ["/cuda/include", "/hip/include"],
        nvmlHeader: "/cuda/include/nvml.h", hipDriverLibrary: "/host/runtime/amdhip64.lib",
        dpcppComponent: "/components/dpcpp", hostComponent: "/components/host", cudaImages: "/images/cuda-images.inc"}},
    input = {runAttempt: suffix,
      llvmDescriptor: {version: 1, tools: {clangCl: "/tools/icx-cl.exe", lldLink: "/tools/lld-link.exe", llvmNm: "/tools/llvm-nm.exe",
        clangResourceInclude: "/tools/resource/include"}, windowsSdk: {sourceRoot: sdkRoot, overlay: at(sdkRoot, "overlay.json"),
        includeRoots: [at(sdkRoot, "msvc/include"), at(sdkRoot, "sdk/include")],
        libraryRoots: [at(sdkRoot, "sdk/lib/ucrt"), at(sdkRoot, "sdk/lib/um"), at(sdkRoot, "msvc/lib"), intelRoot],
        msvcLibraries: [at(sdkRoot, "msvc/lib/libcmt.lib"), ...Object.values(intelLibraries)],
        ucrtLibraries: [at(sdkRoot, "sdk/lib/ucrt/libvcruntime.lib")], importLibraries: [at(sdkRoot, "sdk/lib/um/kernel32.lib")]}},
      io: {kind: "io", runAttempt: suffix, handoff: at(ioRoot, "handoff.json"), handoffSha256: digest, files: {},
        libraries: {sourceInclude: at(ioRoot, "openssl-sdk/include"), ssl: at(ioRoot, "openssl-sdk/lib/libssl.lib"),
          crypto: at(ioRoot, "openssl-sdk/lib/libcrypto.lib"), libuv: at(ioRoot, "io-sdk/lib/uv_a.lib"),
          zlib: at(ioRoot, "io-sdk/lib/zlibstatic.lib"), libuvInclude: at(ioRoot, "io-sdk/include"),
          zlibInclude: at(ioRoot, "io-sdk/include"), zlibGeneratedInclude: at(ioRoot, "io-sdk/include")}},
      hermes: {kind: "hermes", runAttempt: suffix, handoff: at(hermesRoot, "handoff.json"), handoffSha256: digest, files: {},
        hermes: {source: at(hermesRoot, "hermes/source"), build: at(hermesRoot, "hermes/build")},
        intelRuntime: {libRoot: intelRoot, libraries: intelLibraries,
          licenses: {license: at(hermesRoot, "intel-runtime/licenses/LICENSE.rtf")}}},
      hermesAot: {artifacts: {compiler: hermesCompiler}, files: {[hermesCompiler]: digest}},
      tools: {shermaot: hermesCompiler, llvmReadobj: "/tools/llvm-readobj.exe"},
      allowedImports, application};
  return composeWindowsApplicationConfig(input);
}

/** Make same-run external bundle and receipt pins for the production pin admission step. @param {string} suffix */
function pinEnvironment(suffix) {
  const gpuPins = profiles.map((profile, index) => ({profile, bundleSha256: String(index + 1).repeat(64),
    receiptSha256: String(index + 2).repeat(64), configSha256: String(index + 3).repeat(64), handoffSha256: String(index + 4).repeat(64)}));
  return {...Object.fromEntries(pinNames.map(name => [`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`, "a".repeat(64)])),
    MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(gpuPins), GITHUB_RUN_ID: suffix.split("-")[0], GITHUB_RUN_ATTEMPT: "1"};
}

test.describe("Windows full-application reusable job", {skip: process.platform !== "linux" && "Linux workflow-job harness"}, () => {
  test("keeps the public workflow build-only, bounded, exact-artifact and cache-free", () => {
    assert.match(workflow, /^ {2}workflow_call:$/m);
    assert.doesNotMatch(workflow, /^\s+(?:push|pull_request|workflow_dispatch):/m);
    assert.match(workflow, /^ {4}runs-on: ubuntu-24\.04$/m);
    assert.match(workflow, /^ {4}timeout-minutes: 60$/m);
    assert.match(workflow, /now \+ 3300/);
    assert.match(workflow, /MOM_BUILD_JOBS: "2"/);
    assert.doesNotMatch(workflow, /actions\/cache(?:\/|@)|^\s*cache:/m);
    assert.doesNotMatch(workflow, /^ {4}runs-on:.*(?:gpu|self-hosted)/im);
    assert.doesNotMatch(workflow, /^ {4}timeout-minutes: 55$/m);
    for (const name of ["core_bundle_sha256", "core_handoff_sha256", "dpcpp_component_bundle_sha256", "dpcpp_component_manifest_sha256",
      "host_component_bundle_sha256", "host_component_manifest_sha256", "windows_io_bundle_sha256", "windows_io_handoff_sha256",
      "windows_hermes_bundle_sha256", "windows_hermes_handoff_sha256", "linux_hermes_bundle_sha256", "linux_tools_bundle_sha256",
      "linux_tools_handoff_sha256", "dpcpp_inputs_bundle_sha256", "dpcpp_inputs_manifest_sha256", "linux_sources_bundle_sha256",
      "hip_headers_bundle_sha256", "hip_headers_handoff_sha256", "cuda_images_bundle_sha256", "gpu_handoffs_json"]) {
      assert.match(workflow, new RegExp("^      " + name + ":(?:\\n        description:[^\\n]+)?\\n        required: true$", "m"), name);
    }
    assert.match(workflow, /executable_artifact_name: \$\{\{ steps\.seal\.outputs\.executable_artifact_name \}\}/);
    assert.match(workflow, /build_handoff_artifact_name=\$\{evidenceName\}\.tar/);
    const runtimeUpload = workflow.match(/- name: Upload the raw runtime executable only\n([\s\S]*?)(?=\n {6}- name:)/)?.[1] || "",
      evidenceUpload = workflow.match(/- name: Upload compact build receipt and config evidence\n([\s\S]*?)(?=\n?$)/)?.[1] || "";
    assert.match(runtimeUpload, /name: \$\{\{ steps\.seal\.outputs\.executable_artifact_name \}\}/);
    assert.match(runtimeUpload, /path: \$\{\{ steps\.build\.outputs\.executable \}\}/);
    assert.match(runtimeUpload, /archive: true/);
    assert.match(runtimeUpload, /compression-level: 0/);
    assert.match(evidenceUpload, /name: \$\{\{ steps\.seal\.outputs\.build_handoff_artifact_name \}\}/);
    assert.match(evidenceUpload, /path: \$\{\{ runner\.temp \}\}\/native-windows-application-build-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}\.tar/);
    assert.match(evidenceUpload, /archive: false/);
  });

  test("production pin/build/seal heredocs are valid and call the existing readers/composers", () => {
    const pin = nodeBody("Validate every external bundle and receipt pin"),
      build = nodeBody("Restore handoffs, compose the current SDK descriptor, and build"),
      seal = nodeBody("Seal the raw executable and compact build evidence");
    for (const [name, body] of [["pin", pin], ["build", build], ["seal", seal]]) {
      const checked = spawnSync(process.execPath, ["--check"], {input: body, encoding: "utf8", timeout: 30000});
      assert.equal(checked.status, 0, `${name}: ${checked.stderr || checked.stdout}`);
    }
    for (const operation of ["windowsNativeCoreHandoffInputs", "windowsGpuHandoffInputs", "windowsControllerHandoffInputs",
      "windowsDpcppComponentInputs", "windowsHostComponentInputs", "rocmHipHeaderInputs", "nativeHermesInputs",
      "linuxLlvmToolHandoffInputs", "windowsGpuDpcppManifest", "ci.auditDpcppCudaInputs", "ci.verifyLinuxSources",
      "ci.auditLinuxNativeSdkDescriptorCudaImages", "ci.buildWindowsHostSdkSourcesCi()", "ci.composeWindowsLlvmSdkDescriptor",
      "composeWindowsApplicationConfig", "npm, \"run\", \"build:release\""]) {assert.ok(build.includes(operation), operation);}
    assert.match(build, /fileHash\(row\.file\) !== row\.pin/);
    assert.match(build, /fileHash\(row\.file\) !== row\.pin\)\s*\{/);
    assert.match(seal, /ci\.packState\(archive, "\/tmp", \[evidenceName\]/);
    assert.match(seal, /executable_artifact_name=native-windows-executable-\$\{suffix\}/);
  });

  test("full application admits the maintained ordinary Windows DLL set but no application DLLs", () => {
    const build = nodeBody("Restore handoffs, compose the current SDK descriptor, and build"),
      match = build.match(/allowedImports = \[([\s\S]*?)\],\s*config =/);
    assert.ok(match, "the production build declares an explicit Windows import policy");
    const selected = [...(match?.[1] || "").matchAll(/"([^"\n]+\.dll)"/g)].map(row => row[1]);
    assert.deepEqual(selected, ordinaryWindowsImports);
    for (const name of ["bcrypt.dll", "crypt32.dll", "dnsapi.dll", "iphlpapi.dll", "userenv.dll", "ws2_32.dll"]) {
      assert.ok(selected.includes(name), `the full app includes ${name} for ordinary OS services`);
    }
    const config = composeFixture(`${Date.now()}${process.pid}-1`, selected);
    assert.deepEqual(config.windowsSdk.allowedImports, ordinaryWindowsImports);
    assert.throws(() => composeFixture(`${Date.now()}${process.pid}-2`, [...selected, "mom.exe.dll"]), /ordinary system DLLs/);
  });

  test("the production build heredoc restores the planned cohort and invokes the public release command", t => {
    const temp = runnerTemp(t), runId = `${Date.now()}${process.pid}`, suffix = `${runId}-1`, deadline = String(Math.floor(Date.now() / 1000) + 900),
      coreName = `native-windows-core-${suffix}`, dpcppComponentName = `native-windows-dpcpp-static-${suffix}`,
      hostComponentName = `native-windows-host-component-${suffix}`, ioName = `native-windows-controller-io-${suffix}`,
      windowsHermesName = `native-windows-controller-hermes-${suffix}`, linuxHermesSource = `mom-native-hermes-prerequisites-${suffix}`,
      linuxHermesSdk = `mom-native-hermes-sdk-${suffix}`, linuxToolsName = `native-linux-llvm-tools-${suffix}`,
      dpcppInputsName = `mom-native-dpcpp-cuda-inputs-${suffix}`, linuxSourcesName = `mom-native-linux-source-prerequisites-${suffix}`,
      hipHeadersName = `native-rocm-hip-headers-${suffix}`, imagesName = `mom-native-cuda-images-handoff-${suffix}`,
      restoreRoots = [coreName, dpcppComponentName, hostComponentName, ioName, windowsHermesName, linuxHermesSource,
        linuxHermesSdk, linuxToolsName, dpcppInputsName, linuxSourcesName, hipHeadersName, imagesName,
        ...profiles.map(profile => `mom-native-windows-gpu-${profile}-${suffix}`)];
    for (const root of restoreRoots) {
      const location = path.join(root === ioName || root === windowsHermesName || root === linuxToolsName ? temp : "/tmp", root);
      assert.equal(fs.existsSync(location), false, "fixture restore root is new: " + root);
      t.after(() => {if (fs.existsSync(location)) {fs.rmSync(location, {recursive: true, force: true});}});
    }
    const manifestText = (/** @type {string} */ name) => JSON.stringify({fixture: name}) + "\n",
      dpcppManifest = manifestText("dpcpp"), hostManifest = manifestText("host"),
      digest = (/** @type {string} */ value) => require("node:crypto").createHash("sha256").update(value).digest("hex"),
      dpcppManifestSha256 = digest(dpcppManifest), hostManifestSha256 = digest(hostManifest),
      artifactFiles = /** @type {Array<[string,string]>} */ ([
        ["core", `${coreName}.tar`], ["dpcpp-component", `${dpcppComponentName}.tar`], ["host-component", `${hostComponentName}.tar`],
        ["io", `${ioName}.tar`], ["windows-hermes", `${windowsHermesName}.tar`], ["linux-hermes", `native-linux-hermes-sdk-${suffix}.tar`],
        ["linux-tools", `${linuxToolsName}.tar`], ["dpcpp-inputs", `native-linux-dpcpp-cuda-inputs-${suffix}.tar`],
        ["linux-sources", `native-linux-sdk-state-${suffix}-sources.tar`], ["hip-headers", `native-rocm-hip-headers-${suffix}.tar`],
        ["cuda-images", `native-linux-cuda-images-${suffix}.tar`], ...profiles.map(profile => ["gpu", `native-windows-gpu-${profile}-${suffix}.tar`]),
      ]), artifactHashes = /** @type {Record<string,string>} */ ({});
    for (const [directory, name] of artifactFiles) {
      const file = path.join(temp, "incoming", directory, name);
      fs.mkdirSync(path.dirname(file), {recursive: true, mode: 0o700});
      fs.writeFileSync(file, `bounded fixture archive: ${name}\n`, {flag: "wx", mode: 0o444});
      artifactHashes[name] = require("../scripts/build-source-inputs").fileHash(file);
    }
    const output = path.join(temp, `native-windows-application-output-${suffix}`),
      configFile = path.join(temp, `native-windows-application-config-${suffix}.json`), recordFile = path.join(temp, "build-fixture-record.json"),
      githubOutput = path.join(temp, "github-output"), summary = path.join(temp, "step-summary"),
      hostComponentRoot = path.join("/tmp", hostComponentName),
      hipRoot = path.join("/tmp", hipHeadersName), hipInclude = path.join(hipRoot, "include"),
      cudaInclude = path.join("/tmp", dpcppInputsName, "cuda/include"),
      hipDriverLibrary = path.join(hostComponentRoot, "runtime/amdhip64.lib"),
      pins = Object.fromEntries(pinNames.map(name => [`MOM_NATIVE_WINDOWS_APPLICATION_${name}_SHA256`, "a".repeat(64)]));
    fs.writeFileSync(githubOutput, "", {flag: "wx"}); fs.writeFileSync(summary, "", {flag: "wx"});
    Object.assign(pins, {
      MOM_NATIVE_WINDOWS_APPLICATION_CORE_BUNDLE_SHA256: artifactHashes[`${coreName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_DPCPP_COMPONENT_BUNDLE_SHA256: artifactHashes[`${dpcppComponentName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_DPCPP_COMPONENT_MANIFEST_SHA256: dpcppManifestSha256,
      MOM_NATIVE_WINDOWS_APPLICATION_HOST_COMPONENT_BUNDLE_SHA256: artifactHashes[`${hostComponentName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_HOST_COMPONENT_MANIFEST_SHA256: hostManifestSha256,
      MOM_NATIVE_WINDOWS_APPLICATION_IO_BUNDLE_SHA256: artifactHashes[`${ioName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_HERMES_BUNDLE_SHA256: artifactHashes[`${windowsHermesName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_LINUX_HERMES_BUNDLE_SHA256: artifactHashes[`native-linux-hermes-sdk-${suffix}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_LINUX_TOOLS_BUNDLE_SHA256: artifactHashes[`${linuxToolsName}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_DPCPP_INPUTS_BUNDLE_SHA256: artifactHashes[`native-linux-dpcpp-cuda-inputs-${suffix}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_LINUX_SOURCES_BUNDLE_SHA256: artifactHashes[`native-linux-sdk-state-${suffix}-sources.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_HIP_HEADERS_BUNDLE_SHA256: artifactHashes[`native-rocm-hip-headers-${suffix}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_CUDA_IMAGES_BUNDLE_SHA256: artifactHashes[`native-linux-cuda-images-${suffix}.tar`],
      MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(profiles.map(profile => ({profile,
        bundleSha256: artifactHashes[`native-windows-gpu-${profile}-${suffix}.tar`], receiptSha256: "b".repeat(64),
        configSha256: "c".repeat(64), handoffSha256: "d".repeat(64)}))),
    });
    const prelude = `
      const fixtureModule = require("node:module"), fixtureFs = require("node:fs"), fixturePath = require("node:path"), fixtureCrypto = require("node:crypto");
      const fixtureSuffix = process.env.GITHUB_RUN_ID + "-" + process.env.GITHUB_RUN_ATTEMPT, fixtureRunnerTemp = process.env.RUNNER_TEMP,
        fixtureProfiles = ${JSON.stringify(profiles)}, fixtureRestoreCalls = [], fixtureObserved = {compose: null, sdkDescriptor: null, imageAudit: null, buildArgs: null};
      const fixtureHash = file => file === "/opt/intel/oneapi/compiler/2026.2/bin/icx-cl" ? "f".repeat(64) :
        fixtureCrypto.createHash("sha256").update(fixtureFs.readFileSync(file)).digest("hex");
      const fixturePut = (file, value = "fixture\\n") => {fixtureFs.mkdirSync(fixturePath.dirname(file), {recursive: true, mode: 0o700}); fixtureFs.writeFileSync(file, value, {flag: "wx", mode: 0o444});};
      const fakeCi = {
        ciPaths: () => ({suffix: fixtureSuffix, runnerTemp: fixtureRunnerTemp}),
        linuxSourceStatePaths: () => ({sourceName: "mom-native-linux-source-prerequisites-" + fixtureSuffix, sourceTar: fixturePath.join(fixtureRunnerTemp, "native-linux-sdk-state-" + fixtureSuffix + "-sources.tar")}),
        dpcppCudaInputsStatePaths: () => ({name: "mom-native-dpcpp-cuda-inputs-" + fixtureSuffix, sdkTar: fixturePath.join(fixtureRunnerTemp, "native-linux-dpcpp-cuda-inputs-" + fixtureSuffix + ".tar"),
          manifest: fixturePath.join("/tmp", "mom-native-dpcpp-cuda-inputs-" + fixtureSuffix, "inputs.json"), cudaRoot: fixturePath.join("/tmp", "mom-native-dpcpp-cuda-inputs-" + fixtureSuffix, "cuda")}),
        hermesStatePaths: () => ({sourceName: "mom-native-hermes-prerequisites-" + fixtureSuffix, sdkName: "mom-native-hermes-sdk-" + fixtureSuffix,
          sdkRoot: fixturePath.join("/tmp", "mom-native-hermes-sdk-" + fixtureSuffix)}),
        cudaImagesStatePaths: () => ({handoffRoot: fixturePath.join("/tmp", "mom-native-cuda-images-handoff-" + fixtureSuffix)}),
        listState: (file, roots) => roots,
        tarExpandedBytes: () => 1024,
        extractState: (file, destination, roots, allowRoots, timeout) => {
          fixtureRestoreCalls.push({file, destination, roots, allowRoots, timeout});
          for (const root of roots) {fixtureFs.mkdirSync(fixturePath.join(destination, root), {mode: 0o700});}
          for (const root of roots) {
            if (root === "mom-native-dpcpp-cuda-inputs-" + fixtureSuffix) {fixturePut(fixturePath.join(destination, root, "cuda/include/nvml.h")); fixturePut(fixturePath.join(destination, root, "cuda/include/cuda.h"));}
            if (root === "native-windows-dpcpp-static-" + fixtureSuffix) {fixturePut(fixturePath.join(destination, root, "component.json"), ${JSON.stringify(dpcppManifest)});}
            if (root === "native-windows-host-component-" + fixtureSuffix) {
              fixturePut(fixturePath.join(destination, root, "component.json"), ${JSON.stringify(hostManifest)}); fixturePut(fixturePath.join(destination, root, "runtime/amdhip64.lib"));
            }
            if (root === "native-rocm-hip-headers-" + fixtureSuffix) {fixturePut(fixturePath.join(destination, root, "include/hip/hip_runtime_api.h"));}
            if (root === "mom-native-cuda-images-handoff-" + fixtureSuffix) {fixturePut(fixturePath.join(destination, root, "cuda-images.inc"));}
            if (root === "native-linux-llvm-tools-" + fixtureSuffix) {
              for (const name of ["lld-link", "llvm-nm", "llvm-ar", "llvm-objcopy", "llvm-readobj"]) {fixturePut(fixturePath.join(destination, root, "bin", name));}
            }
          }
        },
        verifyLinuxSources: () => {},
        auditDpcppCudaInputs: () => ({}),
        auditLinuxNativeSdkDescriptorCudaImages: state => {fixtureObserved.imageAudit = Object.keys(state).sort(); return {include: fixturePath.join(state.images.handoffRoot, "cuda-images.inc"), includeSha256: "e".repeat(64)};},
        buildWindowsHostSdkSourcesCi: () => ({sdk_receipt_sha256: "a".repeat(64), descriptor_sha256: "b".repeat(64), files_provenance_sha256: "c".repeat(64)}),
        composeWindowsLlvmSdkDescriptor: input => {fixtureObserved.sdkDescriptor = input; return {descriptor: {version: 1, tools: {}, windowsSdk: {}}};},
      };
      const fakeSpawnSync = (command, args, options) => {
        fixtureObserved.buildArgs = {command, args};
        fixtureObserved.compileTransportKeys = Object.keys(options.env).filter(key => key.startsWith("MOM_NATIVE_WINDOWS_APPLICATION_"));
        fixtureObserved.buildOnly = options.env.MOM_AOT_BUILD_ONLY;
        const index = args.indexOf("--native"), config = args[index + 1], output = args[index + 2], binary = fixturePath.join(output, "mom.exe");
        fixtureFs.mkdirSync(output, {mode: 0o700}); fixturePut(binary, "MZ\\n");
        const sha256 = fixtureHash(binary), bytes = fixtureFs.statSync(binary).size, configHash = fixtureHash(config),
          receipt = {target: "x86_64-pc-windows-msvc", nativeRuntime: true, buildOnly: true, executed: false, publicationQualified: false,
            configHash, artifact: {path: binary, bytes, sha256}, application: {graphs: fixtureProfiles.map(profile => ({factory: "mom_native_gpu_" + profile.replaceAll("-", "_")}))}};
        fixturePut(fixturePath.join(output, "result.json"), JSON.stringify(receipt) + "\\n"); fixturePut(fixturePath.join(output, "entrypoint.json"), "{}\\n");
        return {status: 0, signal: null, error: null};
      };
      const fixtureLoaded = fixtureModule._load;
      fixtureModule._load = function(request, parent, isMain) {
        if (request === "./scripts/prepare-native-ci") {return fakeCi;}
        if (request === "./scripts/build-source-inputs") {return {fileHash: fixtureHash, readBuildJson: file => JSON.parse(fixtureFs.readFileSync(file, "utf8"))};}
        if (request === "./scripts/native-linux-llvm-tools") {return {linuxLlvmToolHandoffInputs: (root, expected) => ({toolRoot: fixturePath.join(root, "bin"), lldLink: fixturePath.join(root, "bin/lld-link"),
          llvmNm: fixturePath.join(root, "bin/llvm-nm"), runAttempt: expected.runAttempt})};}
        if (request === "./scripts/native-windows-controller-handoff") {return {windowsControllerHandoffInputs: (root, expected) => ({kind: expected.kind, runAttempt: expected.runAttempt,
          handoff: fixturePath.join(root, "handoff.json"), handoffSha256: expected.handoffSha256, files: {}, ...(expected.kind === "hermes" ? {intelRuntime: {}} : {libraries: {}})})};}
        if (request === "./scripts/native-windows-core-handoff") {return {windowsNativeCoreHandoffInputs: root => ({archive: fixturePath.join(root, "cpu-core.lib")})};}
        if (request === "./scripts/native-windows-dpcpp-component") {return {windowsDpcppComponentInputs: root => ({files: {[fixturePath.join(root, "component.json")]: fixtureHash(fixturePath.join(root, "component.json"))}})};}
        if (request === "./scripts/native-windows-host-component") {return {windowsHostComponentInputs: root => ({files: {[fixturePath.join(root, "component.json")]: fixtureHash(fixturePath.join(root, "component.json"))},
          runtimeInputs: {amdhip64: fixturePath.join(root, "runtime/amdhip64.lib")}})};}
        if (request === "./scripts/native-windows-gpu-handoff") {return {windowsGpuHandoffInputs: (file, expected) => ({profile: expected.profile, runAttempt: expected.runAttempt})};}
        if (request === "./scripts/native-rocm-hip-headers") {return {rocmHipHeaderInputs: root => ({include: fixturePath.join(root, "include"), files: {}})};}
        if (request === "./scripts/native-release-sdk-inputs") {return {nativeHermesInputs: () => ({artifacts: {compiler: fixturePath.join(fixtureRunnerTemp, "fixture-hermes", "shermes")},
          files: {[fixturePath.join(fixtureRunnerTemp, "fixture-hermes", "shermes")]: fixtureHash(fixturePath.join(fixtureRunnerTemp, "fixture-hermes", "shermes"))}})};}
        if (request === "./scripts/native-windows-application-config") {return {composeWindowsApplicationConfig: input => {fixtureObserved.compose = input; return {application: input.application};}};}
        if (request === "./scripts/run-native-windows-gpu-ci") {return {
          windowsGpuHostSdkPins: () => ({resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64), filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)}),
          windowsGpuDpcppManifest: (file, manifestSha256, sourceSha256) => ({source: {tarSha256: sourceSha256}, manifestSha256}),
        };}
        if (request === "node:child_process") {return {spawnSync: fakeSpawnSync};}
        return fixtureLoaded.call(this, request, parent, isMain);
      };
    `, setupScript = prelude + nodeBody("Restore handoffs, compose the current SDK descriptor, and build") +
      "\nfixtureFs.writeFileSync(process.env.FIXTURE_RECORD, JSON.stringify({restoreCalls: fixtureRestoreCalls, observed: fixtureObserved}));";
    const hermesCompilerPath = path.join(temp, "fixture-hermes", "shermes"); fs.mkdirSync(path.dirname(hermesCompilerPath), {recursive: true, mode: 0o700});
    fs.writeFileSync(hermesCompilerPath, "fixture hermes compiler\n", {flag: "wx", mode: 0o444});
    const env = {...process.env, RUNNER_TEMP: temp, GITHUB_RUN_ID: runId, GITHUB_RUN_ATTEMPT: "1", GITHUB_OUTPUT: githubOutput,
      GITHUB_STEP_SUMMARY: summary, FIXTURE_RECORD: recordFile, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
      MOM_NATIVE_WINDOWS_APPLICATION_DEADLINE_EPOCH: deadline, ...pins};
    const result = execNode(setupScript, env);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const record = /** @type {any} */ (JSON.parse(fs.readFileSync(recordFile, "utf8"))),
      restored = /** @type {Array<{destination:string,roots:string[]}>} */ (record.restoreCalls);
    assert.equal(restored.length, 16);
    for (const row of record.restoreCalls) {
      assert.equal(fs.existsSync(row.file), false, "verified transport is discarded before native compilation");
      for (const root of row.roots) {assert.equal(fs.statSync(path.join(row.destination, root)).isDirectory(), true);}
    }
    assert.deepEqual(restored.filter(row => row.roots[0] === ioName || row.roots[0] === windowsHermesName || row.roots[0] === linuxToolsName)
      .map(row => row.destination), Array(3).fill(temp));
    assert.ok(restored.filter(row => ![ioName, windowsHermesName, linuxToolsName].includes(row.roots[0] || ""))
      .every(row => row.destination === "/tmp"));
    assert.deepEqual(record.observed.compose.application.gpuHandoffs.map((/** @type {{profile:string}} */ row) => row.profile).sort(), [...profiles].sort());
    assert.deepEqual(record.observed.compose.application.gpuHandoffs.map((/** @type {{runAttempt:string}} */ row) => row.runAttempt), Array(5).fill(suffix));
    assert.equal(record.observed.compose.application.core, path.join("/tmp", coreName, "cpu-core.lib"));
    assert.deepEqual(record.observed.compose.application.sdk.includeRoots, [cudaInclude, hipInclude]);
    assert.equal(record.observed.compose.application.sdk.nvmlHeader, path.join(cudaInclude, "nvml.h"));
    assert.equal(record.observed.compose.application.sdk.hipDriverLibrary, hipDriverLibrary);
    assert.deepEqual(record.observed.imageAudit, ["images"]);
    const buildArgs = record.observed.buildArgs;
    assert.deepEqual(record.observed.compileTransportKeys, [], "admitted transport pins and changing deadlines must not invalidate compiler caches");
    assert.equal(record.observed.buildOnly, "1");
    assert.equal(buildArgs.command, path.join(repo, "scripts/run-bounded-native-ci.sh"), "the final Windows application link runs inside the existing host memory scope");
    assert.equal(buildArgs.args[0], fs.realpathSync("/usr/bin/timeout"), "the existing aggregate child deadline remains inside that scope");
    assert.equal(buildArgs.args[buildArgs.args.indexOf("npm") + 1], "run");
    assert.deepEqual(buildArgs.args.slice(buildArgs.args.indexOf("--native") - 4), ["npm", "run", "build:release", "--", "--native", configFile, output]);
    assert.equal(fs.existsSync(path.join(output, "mom.exe")), true);
  });

  test("the production pin helper accepts exactly five complete same-run profiles", t => {
    const temp = runnerTemp(t), output = path.join(temp, "github-output");
    fs.writeFileSync(output, "", {flag: "wx"});
    const suffix = `${Date.now()}-1`, base = {...process.env, ...pinEnvironment(suffix), GITHUB_OUTPUT: output}, body = nodeBody("Validate every external bundle and receipt pin");
    const accepted = execNode(body, base);
    assert.equal(accepted.status, 0, accepted.stderr || accepted.stdout);
    assert.match(fs.readFileSync(output, "utf8"), new RegExp(`run_attempt=${suffix}`));

    fs.writeFileSync(output, "", {flag: "w"});
    const duplicated = JSON.parse(String(base.MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON));
    duplicated[4].profile = duplicated[0].profile;
    const wrongCohort = execNode(body, {...base, MOM_NATIVE_WINDOWS_APPLICATION_GPU_HANDOFFS_JSON: JSON.stringify(duplicated)});
    assert.notEqual(wrongCohort.status, 0);
    assert.equal(fs.readFileSync(output, "utf8"), "");

    const malformedPin = execNode(body, {...base, MOM_NATIVE_WINDOWS_APPLICATION_CORE_BUNDLE_SHA256: "bad"});
    assert.notEqual(malformedPin.status, 0);
    assert.equal(fs.readFileSync(output, "utf8"), "");
  });

  test("the actual seal helper refuses missing or expired aggregate deadlines", t => {
    const temp = runnerTemp(t), runId = `${Date.now()}${process.pid}`, attempt = "1", suffix = `${runId}-${attempt}`,
      body = nodeBody("Seal the raw executable and compact build evidence"),
      buildRoot = path.join(temp, `native-windows-application-output-${suffix}`),
      configFile = path.join(temp, `native-windows-application-config-${suffix}.json`),
      executable = path.join(buildRoot, "mom.exe"), resultFile = path.join(buildRoot, "result.json"),
      entrypointFile = path.join(buildRoot, "entrypoint.json"), evidenceRoot = path.join("/tmp", `native-windows-application-build-${suffix}`),
      archive = path.join(temp, `native-windows-application-build-${suffix}.tar`),
      base = {...process.env, RUNNER_TEMP: temp, GITHUB_RUN_ID: runId, GITHUB_RUN_ATTEMPT: attempt,
        BUILD_OUTPUT: buildRoot, CONFIG_FILE: configFile, EXECUTABLE: executable, RESULT_FILE: resultFile, ENTRYPOINT_FILE: entrypointFile};
    assert.equal(fs.existsSync(evidenceRoot), false); assert.equal(fs.existsSync(archive), false);
    const missing = execNode(body, base);
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /aggregate deadline is invalid or expired/);
    const expired = execNode(body, {...base, MOM_NATIVE_WINDOWS_APPLICATION_DEADLINE_EPOCH: "1"});
    assert.notEqual(expired.status, 0);
    assert.match(expired.stderr, /aggregate deadline is invalid or expired/);
    assert.equal(fs.existsSync(evidenceRoot), false); assert.equal(fs.existsSync(archive), false);
  });

  test("the public config composer and actual seal heredoc produce only the raw executable plus compact evidence", t => {
    const {fileHash} = require("../scripts/build-source-inputs"), ci = require("../scripts/prepare-native-ci"), temp = runnerTemp(t),
      suffix = `${Date.now()}${process.pid}-1`, config = composeFixture(suffix), buildRoot = path.join(temp, `native-windows-application-output-${suffix}`),
      configFile = path.join(temp, `native-windows-application-config-${suffix}.json`), executable = path.join(buildRoot, "mom.exe"),
      resultFile = path.join(buildRoot, "result.json"), entrypointFile = path.join(buildRoot, "entrypoint.json"),
      evidenceName = `native-windows-application-build-${suffix}`, evidenceRoot = path.join("/tmp", evidenceName),
      archive = path.join(temp, evidenceName + ".tar"), output = path.join(temp, "github-output"), summary = path.join(temp, "step-summary");
    assert.equal(fs.existsSync(buildRoot), false); assert.equal(fs.existsSync(evidenceRoot), false);
    t.after(() => {if (fs.existsSync(evidenceRoot)) {fs.rmSync(evidenceRoot, {recursive: true, force: true});}});
    fs.mkdirSync(buildRoot, {mode: 0o700});
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o444});
    fs.writeFileSync(executable, Buffer.from("MZ\0fixture-only\n"), {flag: "wx", mode: 0o555});
    const executableSha256 = fileHash(executable), configSha256 = fileHash(configFile), executableBytes = fs.statSync(executable).size,
      result = {target: "x86_64-pc-windows-msvc", nativeRuntime: true, buildOnly: true, executed: false, publicationQualified: false,
        configHash: configSha256, artifact: {path: executable, bytes: executableBytes, sha256: executableSha256},
        application: {graphs: profiles.map(profile => ({factory: "mom_native_gpu_" + profile.replaceAll("-", "_")}))}},
      resultText = JSON.stringify(result, null, 2) + "\n";
    fs.writeFileSync(resultFile, resultText, {flag: "wx", mode: 0o444});
    const resultSha256 = fileHash(resultFile), entrypoint = {version: 1, target: "win32", configFile, configHash: configSha256,
      source: {path: executable, sha256: executableSha256}, artifact: result.artifact, producerReceipt: {path: resultFile, sha256: resultSha256},
      buildOnly: true, publicationQualified: false};
    fs.writeFileSync(entrypointFile, JSON.stringify(entrypoint, null, 2) + "\n", {flag: "wx", mode: 0o444});
    fs.writeFileSync(output, "", {flag: "wx"}); fs.writeFileSync(summary, "", {flag: "wx"});
    const sealed = execNode(nodeBody("Seal the raw executable and compact build evidence"), {...process.env, RUNNER_TEMP: temp,
      GITHUB_RUN_ID: suffix.split("-")[0], GITHUB_RUN_ATTEMPT: "1", GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary,
      MOM_NATIVE_WINDOWS_APPLICATION_DEADLINE_EPOCH: String(Math.floor(Date.now() / 1000) + 300),
      BUILD_OUTPUT: buildRoot, CONFIG_FILE: configFile, EXECUTABLE: executable, RESULT_FILE: resultFile, ENTRYPOINT_FILE: entrypointFile});
    assert.equal(sealed.status, 0, sealed.stderr || sealed.stdout);
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
    const handoff = /** @type {{runAttempt:string,artifact:{sha256:string},gpuHandoffs:Array<{profile:string}>}} */ (
      JSON.parse(fs.readFileSync(path.join(restored, evidenceName, "handoff.json"), "utf8")));
    assert.equal(handoff.runAttempt, suffix);
    assert.equal(handoff.artifact.sha256, executableSha256);
    assert.equal(handoff.gpuHandoffs.length, profiles.length);
    assert.deepEqual(handoff.gpuHandoffs.map(row => row.profile).sort(), [...profiles].sort());
    assert.match(workflow, /path: \$\{\{ steps\.build\.outputs\.executable \}\}[\s\S]*?archive: false/);
  });
});
