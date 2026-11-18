"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {createHash} = require("node:crypto");
const test = require("node:test");
const vm = require("node:vm");
const {extractState, packState, listState, safeRelative, tarExpandedBytes, linuxNativeSdkDiskPreflight} = require("../scripts/native-ci-tar"),
  {ciPaths, hermesStatePaths, assertHermesFetchProvenance, acppStatePaths, assertAcppFetchProvenance, assertAcppPreparedSource, linuxSourceStatePaths, assertSdkFetchProvenance, assertPreparedSdkPaths, linuxSourceFetchArgs, linuxSourcePreparationPlan, rocmParserSource, assertRocmParserReceipt, verifyRocmParserSource, rocmSourceStatePaths, rocmSourceFetchArgs, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, dpcppCudaInputPins, gnuHeadersStatePaths, buildWindowsGnuHeaders, zstdStatePaths, openmpStatePaths} = require("../scripts/native-ci-linux-sources"),
  {admitLinuxLlvmOutput, linuxLlvmConfigureCommand} = require("../scripts/native-ci-linux-llvm"),
  {ioFinalTar, ioOpenSslBuildCommands, hermesBuildCommand, assertHermesSdkFingerprints, zstdBuildCommand, auditZstdSdk, openmpBuildCommand} = require("../scripts/native-ci-linux-support"),
  {ioOpenSslChildTimeout, linuxNativeComponentChildTimeout} = require("../scripts/native-ci-job"),
  {rocmComgrStatePaths, rocmHostStatePaths, rocmHostBuildConfig, rocmResourcesConfig, rocmComgrBuildCommands, buildRocmComgrFromInputs, rocmClrHipamdTreeish, assembleRocmHipHeaders, hipHeaderConfigureCommand, provisionRocmHostPackages, admitRocmParserInstallation, rocmHostNoticeRows, auditRocmHostNoticeSources, acppCudaHipSdkStatePaths, acppCudaHipSdkBuildCommand, assertAcppCudaHipSdk, buildAcppCudaHipSdk} = require("../scripts/native-ci-linux-rocm"),
  {copyDpcppBuildInputs, copyCudaBuildInputs, cudaToolkitProbeCMakeLists, cudaToolkitProbeRoot, probeCudaToolkit, preparedDpcppCudaSourceOrigin, dpcppCudaInputsManifest, auditDpcppCudaInputs} = require("../scripts/native-dpcpp-cuda-inputs"),
  {auditGnuHeaders, capturePackageCopyright} = require("../scripts/prepare-native-gnu-headers"),
  {linuxJitResourcesStatePaths, linuxJitResourceBuildPlan, linuxJitResourceProfile, buildLinuxJitResourceTree, assertLinuxJitResourceManifest, auditLinuxJitResourceTree, linuxJitBootstrapOverlayPlan, seedLinuxJitBootstrapOverlay, auditLinuxJitBootstrapOverlay, linuxJitSdkStatePaths, linuxJitSdkArtifactPhase, linuxJitSdkInputManifest, linuxJitSdkBuildPlan, restoreLinuxJitSdkInputs, auditLinuxJitSdkConfigureArtifacts, linuxJitRuntimeProofPaths, packLinuxJitRuntimeProofFiles, extractLinuxJitRuntimeProof} = require("../scripts/native-ci-linux-jit"),
  {linuxNativeSdkDescriptorStatePaths, linuxNativeSdkGnuNoticeSources, linuxNativeSdkRocmNoticeSources, auditLinuxNativeSdkDescriptorInputArchive, auditLinuxNativeSdkDescriptorLineage, linuxNativeGpuProfiles, linuxNativeComponentProfiles, linuxNativeComponentStatePaths, linuxNativeAssemblyComponents, linuxNativeAssemblyConfig, writeLinuxNativeComponentManifest, auditLinuxNativeComponentTree, linuxNativeComponentArchivePins, auditLinuxNativeComponentArchive, partitionLinuxNativeDownloads, restoreLinuxNativeDescriptorHandoff, linuxNativeApplicationStatePaths, linuxNativeApplicationBuildCommand, auditLinuxNativeApplicationEntrypoint, buildLinuxNativeSdkDescriptor} = require("../scripts/native-ci-linux-release"),
  {linuxNativeSdkLldPath, linuxNativeSdkDescriptorConfig} = require("../scripts/native-linux-sdk-config"),
  {nvrtcStatePaths, cudaImagesStatePaths, cudaImageWorkerRecipe, cudaNoticeSourcePlan, retainCudaNoticeSources, auditCudaNoticeFiles, auditCudaNoticeSources, packNvrtcVerification} = require("../scripts/native-ci-linux-cuda"),
  {windowsNativeCoreStatePaths, windowsNativeCoreDescriptor, windowsNativeCoreDeadlineBudget, auditWindowsNativeCoreOutput, packWindowsNativeCoreHandoff} = require("../scripts/native-ci-windows-core"),
  {windowsHostSdkStatePaths, windowsHostSdkDeadlineBudget, windowsHostSdkPackageMembers, validateWindowsLlvmLibraryPolicy, auditWindowsLlvmToolFile, mapWindowsLlvmSdkLibraries, composeWindowsLlvmSdkDescriptor, composeLocalWindowsLlvmSdkDescriptor, windowsNativeLlvmDescriptorCiStatePaths, windowsNativeLlvmDescriptorCiPreflight, buildWindowsNativeLlvmDescriptorCi, windowsNativeLlvmCiStatePaths, windowsNativeLlvmCiRunDeadlineBudget, windowsNativeLlvmCiDeadlineBudget, windowsNativeLlvmCiDeadlineRecord, windowsNativeLlvmCiPreflight, buildWindowsNativeLlvmCi} = require("../scripts/native-ci-windows-llvm"),
  {windowsGpuHandoffStatePaths, windowsGpuProfileHandoffPlan, windowsGpuDeadlineBudget, windowsGpuProfileCiPreflight, packWindowsGpuHandoff, buildWindowsGpuProfileCi} = require("../scripts/native-ci-windows-gpu"),
  {runStage} = require("../scripts/run-native-ci"),
  {windowsControllerStatePaths, windowsControllerDeadlineBudget, windowsControllerTools, windowsControllerIoBuildCommands, windowsControllerHermesBuildCommand, windowsControllerPackageInventory, windowsControllerIntelRuntimeEntries, verifyWindowsHermesSourceHandoff, packWindowsControllerHandoff} = require("../scripts/native-ci-windows-controller");
const {ioArchiveUrls, gitSourceComponents, sourceRecipe, origins, zstdArchiveCatalog, zstdSourceRecipe, windowsHostSdkCatalog} = require("../scripts/fetch-native-sdk-sources");
const {sdkComponents} = require("../scripts/prepare-native-sdk-sources");
const gnuHeaderProducer = require("../scripts/prepare-native-gnu-headers");
const jitBaseProducer = require("../scripts/prepare-native-jit-base");
const {ioPins} = require("../scripts/build-native-io");
const {opensslPin} = require("../scripts/build-native-openssl");
const {zstdPin} = require("../scripts/build-native-zstd");
const {hermesPins} = require("../scripts/build-native-hermes");
const {archiveName: windowsLlvmArchiveName, archiveSha256: windowsLlvmArchiveSha256} = require("../scripts/native-llvm-sdk");
const windowsIntelRuntimeCatalog = require("../runtime/windows-intel-runtime.json");
const {syclRevision} = require("../scripts/native-sycl-inputs");
const {pins: dpcppPins} = require("../scripts/native-dpcpp-pins");
const {nvrtcPins} = require("../scripts/probe-static-nvrtc");
/** @typedef {ReturnType<typeof import("../scripts/prepare-native-nvrtc-sdk").nvrtcSdkInputs>} NvrtcSdkInputs */
/** @typedef {{path:string,sha256:string}} CudaNoticeFixtureReceipt */
/** @typedef {{component:string,version:string,sourcePath:string,file:string,sha256:string,bytes:number,sourceReceipt:CudaNoticeFixtureReceipt}} CudaNoticeFixtureSource */
/** @typedef {{format:"mom-cuda-notice-sources-v1",scope:"CUDA/NVRTC/CUTLASS source license bodies only",completeNativeNotices:false,
 * sourceArchive:{version:string,url:string,sha256:string,member:string,installer:CudaNoticeFixtureReceipt,archiveMarker:CudaNoticeFixtureReceipt},
 * sources:CudaNoticeFixtureSource[]}} CudaNoticeFixturePlan */
const {hiprtcHeaders} = require("../scripts/build-native-rocm-resources");
const {staticSyclSdkPlan} = require("../scripts/build-native-sycl-sdk");
const {nativeSdkConfig} = require("../scripts/build-native-release");
const {nativeSourceInputs, fileHash} = require("../scripts/build-source-inputs");
const {coffObject} = require("./common/native-windows-component");

const root = path.resolve(__dirname, "..");
const linuxFixedTmpSkip = process.platform === "linux" ? false : "public CI state roots are fixed under Linux /tmp";
const runtimeScript = path.join(root, "scripts/ci/linux-jit-runtime.sh");
// These producer fixtures exercise Linux absolute roots, /usr/bin/tar, or POSIX execute bits.
const linuxOnlySkip = process.platform !== "linux" && "Linux-specific path/archive fixture";

test.describe("Native release runtime workflow", () => {
  test("runtime input commands admit only the selected executable on its runner", t => {
    const {admitCandidate, admitBinary, runtimeLocation} = require("../scripts/ci/runtime-input");
    const temp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-runtime-validator-")));
    t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
    const runnerPlatform = process.platform === "win32" ? "win32" : process.platform === "linux" ? "linux" : "unsupported";
    assert.notEqual(runnerPlatform, "unsupported", "the reusable runtime validator tests run on Linux or Windows");
    const mismatchedPlatform = runnerPlatform === "linux" ? "win32" : "linux", artifactPlatform = runnerPlatform === "win32" ? "windows" : "linux";
    const environment = {RUNNER_TEMP: temp, MOM_NATIVE_CI_SUFFIX: "123456-2",
      NATIVE_EXPECTED_SHA256: "a".repeat(64), NATIVE_PLATFORM: runnerPlatform};
    assert.deepEqual(admitCandidate(environment), {artifact_name: `native-${artifactPlatform}-executable-123456-2`,
      directory: path.join(temp, "mom-native-runtime-123456-2")});
    assert.equal(admitCandidate({...environment, NATIVE_VARIANT: "packed"}).artifact_name,
      `native-${artifactPlatform}-packed-executable-123456-2`, "packed qualification cannot silently consume the raw artifact");
    assert.throws(() => admitCandidate({...environment, NATIVE_VARIANT: "other"}), /variant must be raw or packed/);
    assert.throws(() => admitCandidate({...environment, NATIVE_EXPECTED_SHA256: "bad"}), /64 lowercase/);
    assert.throws(() => admitCandidate({...environment, MOM_NATIVE_CI_SUFFIX: "2/../3"}), /MOM_NATIVE_CI_SUFFIX/);
    assert.throws(() => admitCandidate({...environment, NATIVE_PLATFORM: mismatchedPlatform}), /match this Linux\/Windows x64 runner/,
      "the pre-download gate rejects a target that does not match the current runner");

    const output = path.join(temp, "github-output"), cli = spawnSync(process.execPath, ["scripts/ci/runtime-input.js", "candidate"],
      {cwd: root, env: {...process.env, ...environment, GITHUB_OUTPUT: output}, encoding: "utf8", timeout: 10000, windowsHide: true});
    assert.equal(cli.status, 0, cli.stderr || cli.stdout);
    assert.equal(fs.readFileSync(output, "utf8"), `artifact_name=native-${artifactPlatform}-executable-123456-2\n` +
      "directory=" + path.join(temp, "mom-native-runtime-123456-2") + "\n");

    for (const platform of /** @type {Array<"linux" | "win32">} */ (["linux", "win32"])) {
      const runnerTemp = path.join(temp, platform), bytes = Buffer.from("native executable fixture\n"),
        env = {...environment, RUNNER_TEMP: runnerTemp, NATIVE_PLATFORM: platform, NATIVE_EXPECTED_SHA256: createHash("sha256").update(bytes).digest("hex")},
        {directory, binary} = runtimeLocation(env, platform, "raw");
      fs.mkdirSync(directory, {recursive: true});
      if (platform === runnerPlatform) {assert.throws(() => admitCandidate(env), /new absolute runner-temp path/);}
      assert.throws(() => admitBinary(env), /only the expected executable/, "an empty download is rejected");
      fs.writeFileSync(binary, bytes, {mode: 0o600});
      assert.deepEqual(admitBinary(env), {binary});
      if (process.platform === "linux" && platform === "linux") {
        assert.ok(fs.statSync(binary).mode & 0o111, "Linux artifact download restores execute mode only after its bytes match the audit hash");
      }
      assert.throws(() => admitBinary({...env, NATIVE_EXPECTED_SHA256: "0".repeat(64)}), /differs from the explicit build-audit hash/);
      fs.writeFileSync(path.join(directory, "unexpected.txt"), "extra\n");
      assert.throws(() => admitBinary(env), /only the expected executable/, "extra artifact payload files are rejected");
    }
  });
});

test.describe("Reproducible npm CI installs", () => {
  test("lockfile matches the manifest and pins supported platform binaries", () => {
    const manifest = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
    const lock = JSON.parse(fs.readFileSync(path.join(root, "package-lock.json"), "utf8"));
    const lockedRoot = lock.packages[""];
    assert.equal(lock.lockfileVersion, 3);
    assert.equal(lockedRoot.name, manifest.name);
    assert.equal(lockedRoot.version, manifest.version);
    assert.deepEqual(lockedRoot.devDependencies, manifest.devDependencies);

    for (const [name, entry] of Object.entries(lock.packages)) {
      if (name === "") {continue;}
      assert.equal(typeof entry.version, "string", name + " has a pinned version");
      assert.match(entry.resolved, /^https:\/\/registry\.npmjs\.org\//, name + " has an explicit registry tarball");
      assert.match(entry.integrity, /^sha512-[A-Za-z0-9+/]+={0,2}$/, name + " has a SHA-512 integrity value");
    }

    for (const [name, version, platform] of [
      ["node_modules/@esbuild/linux-x64", manifest.devDependencies.esbuild, "linux"],
      ["node_modules/@esbuild/win32-x64", manifest.devDependencies.esbuild, "win32"],
      ["node_modules/@typescript/typescript-linux-x64", manifest.devDependencies.typescript, "linux"],
      ["node_modules/@typescript/typescript-win32-x64", manifest.devDependencies.typescript, "win32"],
    ]) {
      const entry = lock.packages[name];
      assert.equal(entry.version, version, name + " matches its pinned parent package");
      assert.equal(entry.optional, true, name + " is retained as an optional platform package");
      assert.deepEqual(entry.os, [platform]);
      assert.deepEqual(entry.cpu, ["x64"]);
    }
  });
});

test.describe("Native prerequisite acquisition workflow", () => {
  test("passes the configured bounded job count to the actual Linux LLVM configure command", () => {
    const paths = {source: "/tmp/mom-native-prerequisites-123456-2", output: "/tmp/mom-native-llvm21-sdk-123456-2/linux-llvm21-sdk",
        env: {MOM_BUILD_JOBS: "2"}}, command = linuxLlvmConfigureCommand(paths);
    assert.equal(command.script, "scripts/build-native-llvm-sdk.js");
    assert.deepEqual(command.args, ["configure", path.join(paths.source, "llvm-project-21.1.8.src.tar.xz"),
      paths.output, "/usr/bin/cmake", "/usr/bin/ninja", "/opt/intel/oneapi/compiler/2026.2/bin/icx",
      "/opt/intel/oneapi/compiler/2026.2/bin/icpx", "2"]);
    paths.env.MOM_BUILD_JOBS = "4";
    assert.equal(linuxLlvmConfigureCommand(paths).args.at(-1), "4", "the validated alternate CI envelope remains configurable");
  });

  test("CLI usage lists the maintained stages, not LLVM builder phases", () => {
    const result = spawnSync(process.execPath, ["scripts/run-native-ci.js"], {cwd: root, encoding: "utf8", timeout: 10000}),
      listed = result.stderr.slice(result.stderr.indexOf(" stage ") + 7).trim().split("|").map(entry => entry.split(" ")[0]);
    assert.equal(result.status, 1);
    for (const stage of ["sources", "linux-llvm", "linux-llvm-tools", "linux-native-component", "linux-jit-link", "windows-native-gpu"]) {
      assert.ok(listed.includes(stage), stage);
    }
    for (const phase of ["llvm-foundation", "llvm-x86", "clang-sema", "lld", "tools", "configure", "install-audit"]) {assert.ok(!listed.includes(phase), phase);}
  });

  test("installed LLVM admission fails closed when the external handoff pin is absent or malformed", {skip: linuxOnlySkip}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-llvm-handoff-admission-")), output = path.join(root, "linux-llvm21-sdk");
    fs.mkdirSync(output); fs.writeFileSync(path.join(output, "result.json"), "{}\n", {mode: 0o444});
    fs.writeFileSync(path.join(output, "handoff.json"), "{}\n", {mode: 0o444});
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const base = {output, suffix: "123456-2"};
    assert.throws(() => admitLinuxLlvmOutput({...base, env: {}}), /external handoff pin/,
      "a restored handoff cannot fall back to full-state admission when its pin is missing");
    assert.throws(() => admitLinuxLlvmOutput({...base, env: {MOM_NATIVE_LINUX_LLVM_SDK_HANDOFF_SHA256: "bad"}}), /external handoff pin/,
      "malformed external pins are rejected before reading the handoff");
    assert.throws(() => admitLinuxLlvmOutput({...base, env: {MOM_NATIVE_LINUX_LLVM_SDK_HANDOFF_SHA256: "b".repeat(64)}}), /manifest is invalid or belongs to another run/,
      "a well-shaped but incorrect pin reaches the pinned installed-SDK reader and is rejected there");
  });

  test("oneAPI provisioning reuses a validated 2026.2 install and falls back when it is missing", {skip: linuxOnlySkip}, t => {
    const childProcess = require("node:child_process"), modulePath = require.resolve("../scripts/native-ci-linux-sources"), previous = require.cache[modulePath],
      calls = /** @type {Array<{command:string,args:string[],options:{env?:NodeJS.ProcessEnv,timeout?:number}}>} */ ([]),
      compilerRoot = "/opt/intel/oneapi/compiler/2026.2", compilerCxx = compilerRoot + "/bin/icpx",
      compilerLinker = compilerRoot + "/bin/compiler/ld.lld", fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-oneapi-linker-")),
      fixtureCompilerRoot = path.join(fixtureRoot, "compiler", "2026.2"), fixtureLinkerTarget = path.join(fixtureCompilerRoot, "bin/compiler/lld");
    fs.mkdirSync(path.dirname(fixtureLinkerTarget), {recursive: true}); fs.writeFileSync(fixtureLinkerTarget, "fixture linker\n", {mode: 0o755});
    t.after(() => fs.rmSync(fixtureRoot, {recursive: true, force: true}));
    const toolPaths = new Set(["/usr/bin/cmake", "/usr/bin/ninja", "/usr/bin/python3", "/usr/bin/ld"]),
      realStatSync = fs.statSync.bind(fs), realAccessSync = fs.accessSync.bind(fs), realRealpathSync = fs.realpathSync.bind(fs);
    let validationStatus = 0, missingTool = "";
    t.mock.method(childProcess, "spawnSync", (/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{env?:NodeJS.ProcessEnv,timeout?:number}} */ options) => {
      calls.push({command, args, options});
      if (command === compilerCxx) {
        assert.deepEqual(args, ["-print-prog-name=ld.lld"]);
        return /** @type {any} */ ({error: null, signal: null, status: 0, stdout: compilerLinker + "\n", stderr: ""});
      }
      return /** @type {any} */ ({error: null, signal: null, status: command === "sudo" ? 0 : validationStatus});
    });
    // Keep the helper's real path checks while making this fixture independent of runner packages.
    t.mock.method(fs, "statSync", /** @param {import("node:fs").PathLike} file @param {import("node:fs").StatSyncOptions} [options] */ (file, options) => {
      if (toolPaths.has(String(file))) {
        if (String(file) === missingTool) {throw Object.assign(new Error("fixture prerequisite is absent"), {code: "ENOENT"});}
        return realStatSync(process.execPath, options);
      }
      return realStatSync(file, options);
    });
    t.mock.method(fs, "accessSync", /** @param {import("node:fs").PathLike} file @param {number} [mode] */ (file, mode) => {
      if (String(file) === compilerLinker) {return realAccessSync(fixtureLinkerTarget, mode);}
      if (toolPaths.has(String(file))) {
        if (String(file) === missingTool) {throw Object.assign(new Error("fixture prerequisite is absent"), {code: "ENOENT"});}
        return realAccessSync(process.execPath, mode);
      }
      return realAccessSync(file, mode);
    });
    t.mock.method(fs, "realpathSync", /** @param {import("node:fs").PathLike} file @param {any} [options] */ (file, options) => {
      if (String(file) === compilerRoot) {return fixtureCompilerRoot;}
      if (String(file) === compilerLinker) {return fixtureLinkerTarget;}
      return realRealpathSync(file, options);
    });
    delete require.cache[modulePath];
    const ciSources = require("../scripts/native-ci-linux-sources");
    t.after(() => {delete require.cache[modulePath]; if (previous) {require.cache[modulePath] = previous;}});
    ciSources.provisionOneApi(/** @type {any} */ ({env: process.env}), "/var/tmp/mom-dev-native-test-oneapi", 60 * 1000);
    assert.equal(calls.length, 2, "a successful non-root validation queries the pinned linker without sudo or reinstall");
    assert.equal(calls[0]?.command, path.join(root, "scripts/install-dev.sh"));
    assert.deepEqual(calls[0]?.args, ["--component", "oneapi", "--jobs", "4", "--validate-only"]);
    assert.ok((calls[0]?.options.timeout || 0) <= 2 * 60 * 1000);
    assert.equal(calls[1]?.command, compilerCxx); assert.deepEqual(calls[1]?.args, ["-print-prog-name=ld.lld"]);

    calls.length = 0; validationStatus = 1;
    ciSources.provisionOneApi(/** @type {any} */ ({env: process.env}), "/var/tmp/mom-dev-native-test-oneapi", 60 * 1000);
    assert.equal(calls.length, 4, "a missing compiler triggers signed install, privileged validation and the pinned linker query");
    assert.equal(calls[0]?.command, path.join(root, "scripts/install-dev.sh"));
    assert.deepEqual(calls[0]?.args, ["--component", "oneapi", "--jobs", "4", "--validate-only"]);
    assert.deepEqual(calls[1]?.args, ["env", "MOM_BUILD_JOBS=4", "MOM_DEV_WORKSPACE=/var/tmp/mom-dev-native-test-oneapi",
      path.join(root, "scripts/install-dev.sh"), "--component", "oneapi", "--jobs", "4"]);
    assert.deepEqual(calls[2]?.args, [...(calls[1]?.args || []), "--validate-only"]);
    assert.equal(calls[3]?.command, compilerCxx); assert.deepEqual(calls[3]?.args, ["-print-prog-name=ld.lld"]);

    calls.length = 0; validationStatus = 0; missingTool = "/usr/bin/cmake";
    assert.throws(() => ciSources.provisionOneApi(/** @type {any} */ ({env: process.env}), "/var/tmp/mom-dev-native-test-oneapi", 60 * 1000),
      /Missing Linux LLVM build prerequisite: \/usr\/bin\/cmake/);
    assert.equal(calls.length, 1, "missing canonical tools fail after validation without triggering a reinstall");
  });

  test("pins the original I/O and OpenSSL source archives", () => {
    assert.deepEqual(ioArchiveUrls, {
      libuv: "https://dist.libuv.org/dist/v1.51.0/libuv-v1.51.0.tar.gz",
      zlib: "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz",
    });
    assert.equal(ioPins.libuv.sha256, "5f0557b90b1106de71951a3c3931de5e0430d78da1d9a10287ebc7a3f78ef8eb");
    assert.equal(ioPins.zlib.sha256, "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23");
    assert.equal(opensslPin.version, "3.5.9");
    assert.equal(opensslPin.sha256, "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a");
  });

  test("shrinks I/O/OpenSSL children against one job-start deadline and reserves finalization time", () => {
    const now = Date.UTC(2026, 10, 13, 12), epoch = (/** @type {number} */ minutes) => String(Math.floor((now + minutes * 60 * 1000) / 1000));
    assert.equal(ioOpenSslChildTimeout(epoch(50), 15 * 60 * 1000, now, 5 * 60 * 1000), 15 * 60 * 1000);
    assert.equal(ioOpenSslChildTimeout(epoch(50), 45 * 60 * 1000, now + 20 * 60 * 1000, 5 * 60 * 1000), 25 * 60 * 1000);
    assert.equal(ioOpenSslChildTimeout(epoch(50), 45 * 60 * 1000, now + 44 * 60 * 1000, 5 * 60 * 1000), 60 * 1000);
    assert.throws(() => ioOpenSslChildTimeout("", 1000, now), /job-start deadline/);
    assert.throws(() => ioOpenSslChildTimeout("bad", 1000, now), /job-start deadline/);
    assert.throws(() => ioOpenSslChildTimeout(epoch(-1), 1000, now), /deadline expired/);
    assert.throws(() => ioOpenSslChildTimeout(epoch(61), 1000, now), /60-minute job bound/);
  });

  test("SDK producer timeout terminates its grandchild process group", {skip: process.platform !== "linux"}, async t => {
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-process-group-"));
    t.after(() => fs.rmSync(temporary, {recursive: true, force: true}));
    const {runNodeProcessGroup} = require("../scripts/native-ci-job"),
      timeoutPath = fs.realpathSync("/usr/bin/timeout"), version = spawnSync(timeoutPath, ["--version"], {encoding: "utf8", timeout: 30000});
    if (version.error || version.status !== 0 || !((version.stdout || "") + (version.stderr || "")).includes("GNU coreutils")) {
      assert.throws(() => runNodeProcessGroup("unused.js", [], {}, 1000), /requires GNU coreutils timeout/,
        "the production helper must reject non-GNU timeout implementations");
      t.skip("the process-group fixture requires the production GNU Coreutils timeout");
      return;
    }
    const parentScript = path.join(temporary, "parent.js"), childScript = path.join(temporary, "grandchild.js");
    fs.writeFileSync(parentScript, `const {spawn} = require("node:child_process");
const child = spawn(process.execPath, [process.argv[2]], {stdio: "ignore"});
require("node:fs").writeFileSync(process.argv[3], String(child.pid));
setInterval(() => {}, 1000);
`);
    fs.writeFileSync(childScript, "setInterval(() => {}, 1000);\n");
    /** @param {number} pid */
    const processInfo = pid => {
      try {
        const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8"), fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
        return {state: fields[0] || "", ppid: fields[1] || "", pgrp: fields[2] || "", session: fields[3] || ""};
      } catch (error) {if (Reflect.get(Object(error), "code") === "ENOENT") {return null;} throw error;}
    };
    /** @param {number} pid */
    const isRunning = pid => {const info = processInfo(pid); return info !== null && info.state !== "Z";};
    /** @param {number} pid */
    const waitForExit = async pid => {
      const deadline = Date.now() + 2000;
      while (isRunning(pid) && Date.now() < deadline) {await new Promise(resolve => setTimeout(resolve, 25));}
      return processInfo(pid);
    };
    /** @param {number | undefined} pid */
    const stopFixtureProcess = pid => {
      if (pid && isRunning(pid)) {
        try {process.kill(pid, "SIGKILL");} catch (error) {if (Reflect.get(Object(error), "code") !== "ESRCH") {throw error;}}
      }
    };
    let baselinePid, groupedPid;
    try {
      const baselineFile = path.join(temporary, "baseline.pid"), baseline = spawnSync(process.execPath,
        [parentScript, childScript, baselineFile], {encoding: "utf8", timeout: 1000, windowsHide: true});
      assert.equal(Reflect.get(Object(baseline.error), "code"), "ETIMEDOUT", "the original direct spawnSync timeout expires");
      baselinePid = Number(fs.readFileSync(baselineFile, "utf8"));
      assert.ok(Number.isSafeInteger(baselinePid) && baselinePid > 1);
      assert.ok(isRunning(baselinePid), "before: direct spawnSync timeout leaves its producer grandchild alive");

      const groupedFile = path.join(temporary, "grouped.pid");
      assert.throws(() => runNodeProcessGroup(path.relative(root, parentScript), [childScript, groupedFile], {}, 1000),
        /Bounded SDK producer failed: \S*parent\.js \(deadline\)/);
      groupedPid = Number(fs.readFileSync(groupedFile, "utf8"));
      assert.ok(Number.isSafeInteger(groupedPid) && groupedPid > 1);
      const remaining = await waitForExit(groupedPid);
      assert.ok(!remaining || remaining.state === "Z", `after: GNU timeout must terminate the producer grandchild; pid=${groupedPid}, proc=${JSON.stringify(remaining)}`);
    } finally {
      stopFixtureProcess(groupedPid);
      stopFixtureProcess(baselinePid);
    }
  });

  test("seal admits canonical handoff files and appends every hash only after all pass", t => {
    const {seal} = require("../scripts/ci/seal");
    const directory = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-seal-")));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const archive = path.join(directory, "sdk.tar"), result = path.join(directory, "result.json"), output = path.join(directory, "github-output"),
      env = {GITHUB_OUTPUT: output};
    fs.writeFileSync(archive, "Hermes fixture\n"); fs.writeFileSync(result, "{}\n");
    seal(["bundle_sha256=" + archive, "windows_result_sha256=" + result], env);
    const sealed = "bundle_sha256=" + fileHash(archive) + "\nwindows_result_sha256=" + fileHash(result) + "\n";
    assert.equal(fs.readFileSync(output, "utf8"), sealed);
    const alias = path.join(directory, "alias.tar"); fs.linkSync(archive, alias);
    assert.throws(() => seal(["result_sha256=" + result, "bundle_sha256=" + archive], env), /canonical singly-linked/);
    fs.unlinkSync(alias);
    assert.throws(() => seal(["bundle_sha256=" + path.relative(process.cwd(), archive)], env), /canonical singly-linked/);
    assert.throws(() => seal(["bundle_sha256=" + archive, "bundle_sha256=" + result], env), /unique/);
    assert.throws(() => seal(["Bundle=" + archive], env), /unique/);
    assert.throws(() => seal([archive], env), /unique/);
    assert.throws(() => seal([], env), /Usage/);
    assert.throws(() => seal(["bundle_sha256=" + archive], {}), /GITHUB_OUTPUT/);
    fs.writeFileSync(path.join(directory, "empty.tar"), "");
    assert.throws(() => seal(["bundle_sha256=" + path.join(directory, "empty.tar")], env), /canonical singly-linked/);
    if (process.platform !== "win32") {
      fs.symlinkSync(archive, path.join(directory, "link.tar"));
      assert.throws(() => seal(["bundle_sha256=" + path.join(directory, "link.tar")], env), /canonical singly-linked/);
    }
    assert.equal(fs.readFileSync(output, "utf8"), sealed, "a rejected file leaves the step outputs unchanged");
    const cli = spawnSync(process.execPath, ["scripts/ci/seal.js", "bundle_sha256=" + path.join(directory, "missing.tar")],
      {cwd: root, env: {...process.env, GITHUB_OUTPUT: output}, encoding: "utf8", timeout: 10000, windowsHide: true});
    assert.equal(cli.status, 1);
    assert.match(cli.stderr, /^\[native-ci\] /);
  });

  test("GNU target headers have a bounded independent data-only same-run producer", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-gnu-ci-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123-1", runnerTemp: fs.realpathSync(runnerTemp)}, environment = {MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/mom-native-gnu-headers-123-1"},
      state = gnuHeadersStatePaths(paths, environment);
    assert.equal(state.root, environment.MOM_NATIVE_GNU_HEADERS_DIR);
    assert.equal(path.basename(state.sdkTar), "native-linux-gnu-headers-123-1.tar");
    assert.throws(() => gnuHeadersStatePaths(paths, {...environment, MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/stale"}), /exactly match/);
    const windowsTarget = "x86_64-w64-windows-gnu", windowsEnvironment = {MOM_NATIVE_WINDOWS_GNU_HEADERS_DIR: "/tmp/mom-native-windows-gnu-headers-123-1"},
      windowsState = gnuHeadersStatePaths(paths, windowsEnvironment, windowsTarget);
    assert.equal(windowsState.target, windowsTarget);
    assert.equal(windowsState.name, "mom-native-windows-gnu-headers-123-1");
    assert.equal(windowsState.root, windowsEnvironment.MOM_NATIVE_WINDOWS_GNU_HEADERS_DIR);
    assert.equal(path.basename(windowsState.sdkTar), "native-windows-gnu-headers-123-1.tar");
    assert.throws(() => gnuHeadersStatePaths(paths, {...windowsEnvironment, MOM_NATIVE_WINDOWS_GNU_HEADERS_DIR: "/tmp/stale"}, windowsTarget), /exactly match/);
    assert.equal(typeof buildWindowsGnuHeaders, "function");
    assert.equal(capturePackageCopyright, gnuHeaderProducer.capturePackageCopyright);
    assert.deepEqual(gnuHeaderProducer.packageCopyrightAlias("g++-mingw-w64-x86-64-posix"), {
      resolvedSourcePath: "/usr/share/doc/gcc-mingw-w64-base/copyright",
      ownershipPath: "/usr/share/doc/g++-mingw-w64-x86-64-posix",
    });
    assert.equal(gnuHeaderProducer.packageCopyrightAlias("unknown-package"), undefined);
    assert.equal(auditGnuHeaders, gnuHeaderProducer.auditGnuHeaders);
    const run = String(Date.now()) + String(process.pid), sealPaths = {suffix: run + "-1", runnerTemp: paths.runnerTemp},
      sealState = gnuHeadersStatePaths(sealPaths, {MOM_NATIVE_WINDOWS_GNU_HEADERS_DIR: "/tmp/mom-native-windows-gnu-headers-" + run + "-1"}, windowsTarget),
      result = path.join(sealState.root, "result.json"), copyrights = path.join(sealState.root, "copyrights.json"),
      output = path.join(paths.runnerTemp, "github-output");
    assert.equal(fs.existsSync(sealState.root), false, "fixture root must be fresh before it is created");
    fs.mkdirSync(sealState.root);
    t.after(() => fs.rmSync(sealState.root, {recursive: true, force: true}));
    fs.writeFileSync(result, "tiny GNU result\n", {flag: "wx"});
    fs.writeFileSync(copyrights, "tiny notices\n", {flag: "wx"});
    fs.writeFileSync(sealState.sdkTar, "tiny GNU tar fixture\n", {flag: "wx"});
    require("../scripts/ci/seal").seal(["windows_bundle_sha256=" + sealState.sdkTar, "windows_result_sha256=" + result,
      "windows_copyrights_sha256=" + copyrights], {GITHUB_OUTPUT: output});
    assert.deepEqual(fs.readFileSync(output, "utf8").trimEnd().split("\n"), [
      "windows_bundle_sha256=" + fileHash(sealState.sdkTar),
      "windows_result_sha256=" + fileHash(result),
      "windows_copyrights_sha256=" + fileHash(copyrights),
    ], "the matrix seal reads the producer's canonical /tmp root and its RUNNER_TEMP tar");
  });

  test("Windows GNU header admission keeps the known g++ copyright alias exact", {skip: linuxOnlySkip}, t => {
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-gnu-copyright-ci-")), sysroot = path.join(parent, "sysroot"),
      output = path.join(parent, "output"), target = "x86_64-w64-windows-gnu";
    t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
    fs.mkdirSync(sysroot);
    /** @param {string} relative @param {string} content */
    const write = (relative, content) => {
      const file = path.join(sysroot, relative); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, content); return file;
    };
    const common = write("usr/share/mingw-w64/include/windows.h", "common system fixture\n"),
      alias = path.join(sysroot, "usr/x86_64-w64-mingw32/include/windows.h");
    fs.mkdirSync(path.dirname(alias), {recursive: true}); fs.symlinkSync(path.relative(path.dirname(alias), common), alias);
    const cxx = write("usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/vector", "common C++ fixture\n"),
      backward = write("usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/backward/hash.h", "backward fixture\n"),
      targetCxx = write("usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/x86_64-w64-mingw32/bits/config.h", "target C++ fixture\n");
    /** @param {string[]} files */
    const logicalFiles = files => files.map((/** @type {string} */ file) => "/" + path.relative(sysroot, file).split(path.sep).join("/")), packages = [
      {name: "mingw-w64-common", version: "13.0.0-2ubuntu1", architecture: "all", files: logicalFiles([common])},
      {name: "mingw-w64-x86-64-dev", version: "13.0.0-2ubuntu1", architecture: "all", files: logicalFiles([alias])},
      {name: "g++-mingw-w64-x86-64-posix", version: "13.2.0-6ubuntu1+26.1", architecture: "amd64", files: logicalFiles([cxx, backward, targetCxx])},
    ];
    gnuHeaderProducer.prepareWindowsGnuHeaders(sysroot, output, packages);
    const headers = gnuHeaderProducer.verifyWindowsGnuHeaders(output), aliases = gnuHeaderProducer.packageCopyrightAlias,
      notices = packages.map(row => {
        const file = path.join(output, "licenses", row.name, "copyright");
        fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture terms\n");
        return {package: row.name, version: row.version, sourcePath: "/usr/share/doc/" + row.name + "/copyright", path: file,
          sha256: createHash("sha256").update("fixture terms\n").digest("hex"),
          ...(aliases(row.name) ? aliases(row.name) : {})};
      }), index = path.join(output, "copyrights.json");
    assert.equal(headers.files.length, 4);
    fs.writeFileSync(index, JSON.stringify(notices));
    assert.equal(auditGnuHeaders({root: output, target}, target).headers.files.length, 4);
    const gpp = notices[2];
    if (!gpp) {throw new Error("Missing g++ package notice fixture");}
    Object.assign(gpp, {resolvedSourcePath: "/usr/share/doc/other-package/copyright"});
    fs.writeFileSync(index, JSON.stringify(notices));
    assert.throws(() => auditGnuHeaders({root: output, target}, target), /copyright content, package or path changed/);
  });

  test("shared package copyright capture retains owned non-GNU Debian aliases", {skip: linuxOnlySkip}, t => {
    const childProcess = require("node:child_process"), nativeFs = require("node:fs"), parent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-generic-copyright-alias-")),
      body = path.join(parent, "copyright-source"), root = path.join(parent, "output"), packageName = "fixture-mingw-unknown",
      source = "/usr/share/doc/" + packageName + "/copyright", resolved = "/usr/share/doc/unlisted-base/copyright",
      ownership = "/usr/share/doc/" + packageName, terms = "fixture owned copyright\n";
    fs.writeFileSync(body, terms); fs.mkdirSync(root);
    t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
    const original = {realpathSync: nativeFs.realpathSync, statSync: nativeFs.statSync, lstatSync: nativeFs.lstatSync,
      copyFileSync: nativeFs.copyFileSync, openSync: nativeFs.openSync};
    t.mock.method(nativeFs, "realpathSync", (/** @type {string} */ file) => file === source ? resolved : original.realpathSync(file));
    t.mock.method(nativeFs, "statSync", (/** @type {string} */ file) => file === resolved ? original.statSync(body) : original.statSync(file));
    t.mock.method(nativeFs, "lstatSync", (/** @type {string} */ file) => file === ownership ?
      /** @type {any} */ ({isSymbolicLink: () => true}) : original.lstatSync(file));
    t.mock.method(nativeFs, "copyFileSync", (/** @type {string} */ from, /** @type {string} */ to, /** @type {number} */ flags) =>
      original.copyFileSync(from === source ? body : from, to, flags));
    t.mock.method(nativeFs, "openSync", (/** @type {string} */ file, /** @type {string | number} */ flags, /** @type {number} */ mode) =>
      original.openSync(file === source ? body : file, flags, mode));
    t.mock.method(childProcess, "spawnSync", (/** @type {string} */ file, /** @type {string[]} */ args) => {
      assert.equal(file, "/usr/bin/dpkg-query"); assert.deepEqual(args, ["-S", ownership]);
      return /** @type {any} */ ({error: null, signal: null, status: 0, stdout: packageName + ": " + ownership + "\n"});
    });
    const key = require.resolve("../scripts/prepare-native-gnu-headers"), previous = require.cache[key];
    delete require.cache[key];
    const fresh = require("../scripts/prepare-native-gnu-headers");
    t.after(() => {delete require.cache[key]; if (previous) {require.cache[key] = previous;}});
    const captured = fresh.capturePackageCopyright({root}, packageName, "1.0-1");
    assert.equal(captured.resolvedSourcePath, resolved);
    assert.equal(captured.ownershipPath, ownership);
    assert.equal(fs.readFileSync(captured.path, "utf8"), terms);
    assert.equal(fresh.packageCopyrightAlias(packageName), undefined);
  });

  test("GNU header admission rechecks transported copyright bytes and package identity", {skip: process.platform !== "linux"}, t => {
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gnu-copyright-ci-")), include = path.join(parent, "include"), output = path.join(parent, "output");
    t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
    const specs = [["libstdc++-13-dev", ["c++/13/vector", "c++/13/backward/hash.h", "x86_64-linux-gnu/c++/13/bits/config.h"]],
      ["libc6-dev", ["x86_64-linux-gnu/sys/types.h"]], ["linux-libc-dev", ["linux/types.h"]]];
    const packages = specs.map(([name, entries]) => {
      if (typeof name !== "string" || !Array.isArray(entries)) {throw new Error("Invalid test package");}
      const files = entries.map(relative => {
        const file = path.join(include, relative); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture header\n"); return file;
      });
      return {name, version: "1.2.3", architecture: "amd64", files};
    });
    require("../scripts/prepare-native-gnu-headers").prepareLinuxGnuHeaders(include, output, packages);
    const notices = packages.map(row => {
      const file = path.join(output, "licenses", row.name, "copyright");
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture terms\n");
      return {package: row.name, version: row.version, sourcePath: "/usr/share/doc/" + row.name + "/copyright", path: file,
        sha256: createHash("sha256").update("fixture terms\n").digest("hex")};
    });
    const index = path.join(output, "copyrights.json"), save = () => fs.writeFileSync(index, JSON.stringify(notices));
    save(); assert.equal(auditGnuHeaders({root: output}).notices.length, 3);
    fs.writeFileSync(notices[0]?.path || "", "changed terms");
    assert.throws(() => auditGnuHeaders({root: output}), /copyright content, package or path changed/);
    fs.writeFileSync(notices[0]?.path || "", "fixture terms\n");
    if (notices[1]) {notices[1].version = "different";}
    save(); assert.throws(() => auditGnuHeaders({root: output}), /copyright content, package or path changed/);
    if (notices[1]) {notices[1].version = "1.2.3";}
    save(); assert.equal(auditGnuHeaders({root: output}).headers.files.length, 5);
  });

  test("Linux JIT resource stage binds exact same-run inputs and public producer commands", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-resources-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)}, environment = {
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix,
        MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/mom-native-gnu-headers-" + suffix,
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-" + suffix,
      }, state = linuxJitResourcesStatePaths(paths, environment);
    assert.equal(state.root, environment.MOM_NATIVE_LINUX_JIT_RESOURCES_DIR);
    assert.equal(path.basename(state.incomingSourceTar), "native-linux-sdk-state-123456-2-sources.tar");
    assert.equal(path.basename(state.incomingDpcppTar), "native-linux-dpcpp-cuda-inputs-123456-2.tar");
    assert.equal(path.basename(state.incomingGnuTar), "native-linux-gnu-headers-123456-2.tar");
    assert.equal(path.basename(state.sdkTar), "native-linux-jit-resources-123456-2.tar");
    assert.equal(state.bootstrapBuild, path.join(state.bootstrap, "build"));
    assert.throws(() => linuxJitResourcesStatePaths(paths, {...environment,
      MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/stale-jit-resource"}), /exactly match/);

    const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/tmp/compiler-sources/" + name])),
      tools = {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja", cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx",
        cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"}, plan = linuxJitResourceBuildPlan(state, roots, tools, "20260711");
    assert.deepEqual(plan.emitter, {script: "scripts/native-jit-sdk.js", args: ["emitter", state.dpcpp.dpcppRoot, state.emitter,
      path.join(state.gnu.root, "target-headers.json")]});
    assert.equal(plan.configure[0], tools.cmake);
    assert.ok(plan.configure.includes("-DSYCL_HEADERS_ONLY=ON"));
    assert.ok(plan.configure.includes("-DSYCL_JIT_RESOURCE_OVERLAY=" + state.bootstrapOverlay));
    assert.ok(plan.configure.includes("-DSYCL_COMPILER_VERSION=20260711"));
    assert.ok(plan.configure.includes("-DCMAKE_C_COMPILER=" + tools.cc));
    assert.ok(plan.configure.includes("-DCMAKE_CXX_COMPILER=" + tools.cxx));
    assert.ok(plan.configure.includes("-DFETCHCONTENT_FULLY_DISCONNECTED=ON"));
    assert.deepEqual(plan.base, {script: "scripts/prepare-native-jit-base.js", args: ["--sdk", state.dpcpp.dpcppRoot,
      "--cuda", state.dpcpp.cudaRoot, "--build", state.bootstrapBuild, "--sources", state.linux.compilerPreparedReceipt,
      "--dpcpp-git", path.join(state.linux.source, "roots/dpcpp"), "--output", state.base]});
    assert.deepEqual(plan.resources, {script: "scripts/prepare-native-sycl-jit-resources.js", args: ["--profile", state.profile, "--output", state.resources]});

    const headers = {root: state.gnu.root}, noticeRows = ["libstdc++-13-dev", "libc6-dev", "linux-libc-dev"].map(packageName => ({
        package: packageName, path: path.join(state.gnu.root, "licenses", packageName, "copyright"), sha256: "a".repeat(64),
      })), profile = linuxJitResourceProfile(state, {fixture: path.join(state.emitter, "native-kawpow-source"),
        compiler: path.join(state.dpcpp.dpcppRoot, "bin/clang++"), targetHeaders: []}, {headers, notices: noticeRows});
    assert.equal(profile.target, "x86_64-unknown-linux-gnu");
    assert.equal(profile.headerRoots.length, 4);
    assert.equal(profile.syclInclude, path.join(state.base, "base/include"));
    assert.deepEqual(profile.headerRoots.map(row => row.virtual), ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target", "include/mom-system"]);
    assert.deepEqual(profile.notices.map(row => row.source), noticeRows.map(row => path.relative(state.gnu.root, row.path).split(path.sep).join("/")));
    assert.deepEqual(profile.notices.map(row => row.sha256), noticeRows.map(row => row.sha256));
    assert.deepEqual(Object.keys(profile.required).sort(), ["libdevice", "spirv32", "spirv64"]);
  });

  test("Linux JIT resource manifest binds required bitcode and complete base provenance", t => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-resource-manifest-")), baseRoot = path.join(directory, "base");
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    for (const row of Object.values(jitBaseProducer.REQUIRED)) {
      const file = path.join(baseRoot, ...row.virtual.split("/"));
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture bitcode\n");
    }
    const inventory = jitBaseProducer.baseManifest(baseRoot),
      /** @type {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} */
      expectedBaseRows = inventory.files.map(row => ({...row, root: /** @type {"base"} */ ("base"),
        bytes: fs.statSync(path.join(baseRoot, row.source)).size})),
      required = Object.fromEntries(Object.entries(jitBaseProducer.REQUIRED).map(([name, row]) => {
        const baseRow = inventory.files.find(file => file.virtual === row.virtual);
        assert.ok(baseRow, "each pinned device library must be in the actual base inventory");
        return [name, {virtual: row.virtual, sha256: baseRow.sha256}];
      })),
      manifest = {required, files: [...expectedBaseRows,
        {root: "gnu", source: "include/mom-cxx/c++config.h", virtual: "include/mom-cxx/c++config.h", sha256: "b".repeat(64), bytes: 16}]};
    assert.doesNotThrow(() => assertLinuxJitResourceManifest(manifest, required, expectedBaseRows));

    /** @param {(candidate:{required:Record<string,{virtual:string,sha256:string}>,files:Array<{root:string,source:string,virtual:string,sha256:string,bytes:number}>})=>void} mutate */
    const rejectsMutation = mutate => {
      const candidate = /** @type {{required:Record<string,{virtual:string,sha256:string}>,files:Array<{root:string,source:string,virtual:string,sha256:string,bytes:number}>}} */ (
        JSON.parse(JSON.stringify(manifest)));
      mutate(candidate);
      assert.throws(() => assertLinuxJitResourceManifest(candidate, required, expectedBaseRows), /Linux JIT resource manifest/);
    };
    rejectsMutation(candidate => {
      const row = candidate.files.find(item => item.root === "base"); assert.ok(row); row.source = "substituted/source.hpp";
    });
    rejectsMutation(candidate => {candidate.files.splice(candidate.files.findIndex(row => row.root === "base"), 1);});
    rejectsMutation(candidate => {
      const row = candidate.required["libdevice"]; assert.ok(row); row.sha256 = "f".repeat(64);
    });
    rejectsMutation(candidate => {
      const row = candidate.files.find(item => item.root === "gnu"); assert.ok(row); row.root = "unrecognized";
    });
    const substitutedRequired = JSON.parse(JSON.stringify(required)), substitutedManifest = JSON.parse(JSON.stringify(manifest));
    substitutedRequired.libdevice.sha256 = "f".repeat(64); substitutedManifest.required = substitutedRequired;
    assert.throws(() => assertLinuxJitResourceManifest(substitutedManifest, substitutedRequired, expectedBaseRows), /required device libraries/);
  });

  test("local and CI JIT resource trees share the producer without archive or installer work", {skip: linuxOnlySkip}, t => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-shared-jit-tree-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    assert.throws(() => buildLinuxJitResourceTree(
      /** @type {Parameters<typeof buildLinuxJitResourceTree>[0]} */ (/** @type {unknown} */ ({root: directory})),
      {cmake: "/unused/cmake", ninja: "/unused/ninja", cc: "/unused/icx", cxx: "/unused/icpx"}, {}), /refuses an existing output/);
    const state = /** @type {Parameters<typeof linuxJitBootstrapOverlayPlan>[0]} */ (/** @type {unknown} */ ({bootstrapOverlay: path.join(directory, "overlay")})),
      rows = linuxJitBootstrapOverlayPlan(state, {headers: {root: path.join(directory, "gnu")}},
        {root: path.join(directory, "opencl"), revision: "a".repeat(40)});
    assert.equal(rows.length, 5);
    assert.equal(rows[4]?.source, path.join(directory, "opencl/CL"));
    assert.throws(() => linuxJitBootstrapOverlayPlan(state, {headers: {root: "relative"}},
      {root: path.join(directory, "opencl"), revision: "a".repeat(40)}), /admitted GNU/);
    const auditState = /** @type {Parameters<typeof auditLinuxJitResourceTree>[0]} */ (/** @type {unknown} */
        ({base: path.join(directory, "base"), emitter: path.join(directory, "emitter")})),
      tools = {cmake: "/unused/cmake", ninja: "/unused/ninja", cc: "/unused/icx", cxx: "/unused/icpx"};
    fs.mkdirSync(auditState.base); fs.mkdirSync(auditState.emitter);
    fs.mkdirSync(path.join(directory, "redirected-base"));
    fs.symlinkSync(path.join(directory, "redirected-base"), path.join(auditState.base, "base"));
    fs.writeFileSync(path.join(auditState.emitter, "native-kawpow-source"), "fixture");
    assert.throws(() => auditLinuxJitResourceTree(auditState, tools, {}), /canonical real/);
    fs.unlinkSync(path.join(auditState.base, "base")); fs.mkdirSync(path.join(auditState.base, "base"));
    fs.renameSync(path.join(auditState.emitter, "native-kawpow-source"), path.join(directory, "redirected-fixture"));
    fs.symlinkSync(path.join(directory, "redirected-fixture"), path.join(auditState.emitter, "native-kawpow-source"));
    assert.throws(() => auditLinuxJitResourceTree(auditState, tools, {}), /canonical real/);
  });

  test("Linux JIT resource artifact rejects stale or substituted same-run input roots", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-input-tars-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)}, environment = {
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix,
        MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/mom-native-gnu-headers-" + suffix,
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-" + suffix,
      }, state = linuxJitResourcesStatePaths(paths, environment), wrongSuffix = "123456-1";
    const archiveDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-input-archives-"));
    t.after(() => fs.rmSync(archiveDirectory, {recursive: true, force: true}));
    /** @type {Array<[string,string,string]>} */ const cases = [
      [state.linux.sourceName, "mom-native-linux-source-prerequisites-" + wrongSuffix, "wrong-sources.tar"],
      [state.dpcpp.name, "mom-native-dpcpp-cuda-inputs-" + wrongSuffix, "wrong-dpcpp.tar"],
      [state.gnu.name, "mom-native-gnu-headers-" + wrongSuffix, "wrong-gnu.tar"],
    ];
    for (const [expected, wrong, archive] of cases) {
      const base = path.join(runnerTemp, wrong), file = path.join(base, "marker");
      fs.mkdirSync(base); fs.writeFileSync(file, "stale run attempt\n");
      const tarFile = path.join(archiveDirectory, archive); packState(tarFile, runnerTemp, [wrong]);
      assert.throws(() => listState(tarFile, [expected]), /unexpected state roots/);
    }
  });

  test("Linux JIT resource seal binds the unchanged tar, result and build-only manifest", {skip: linuxOnlySkip}, t => {
    const {sealLinuxJitResources, assertLinuxJitResourceResult} = require("../scripts/ci/linux-sdk");
    const runnerTemp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-seal-")));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", name = "mom-native-linux-jit-resources-" + suffix, base = path.join(runnerTemp, "state"),
      linuxJitResourcesRoot = path.join(base, name), manifestFile = path.join(linuxJitResourcesRoot, "resources", "manifest.json"),
      resultFile = path.join(linuxJitResourcesRoot, "ci-result.json"), archive = path.join(runnerTemp, "native-linux-jit-resources-" + suffix + ".tar"),
      target = "x86_64-unknown-linux-gnu", manifest = {target, publicationQualified: false, files: [{path: "base/libdevice.bc"}]};
    fs.mkdirSync(path.dirname(manifestFile), {recursive: true});
    fs.writeFileSync(manifestFile, JSON.stringify(manifest) + "\n");
    const manifestSha256 = fileHash(manifestFile), result = {version: 1, target, buildOnly: true, executed: false, publicationQualified: false,
      resources: {manifest: manifestFile, manifestSha256, files: 1}};
    fs.writeFileSync(resultFile, JSON.stringify(result) + "\n");
    packState(archive, base, [name]);
    const paths = {suffix, runnerTemp, linuxJitResourcesRoot};
    assert.deepEqual(sealLinuxJitResources(paths), {resources_bundle_sha256: fileHash(archive),
      resources_manifest_sha256: manifestSha256, resources_result_sha256: fileHash(resultFile)});
    for (const changed of [{...result, executed: true}, {...result, publicationQualified: true}, {...result, target: "x86_64-w64-windows-gnu"},
      {...result, resources: {...result.resources, files: 2}}, {...result, resources: {...result.resources, manifestSha256: "0".repeat(64)}},
      {...result, resources: {...result.resources, manifest: path.join(runnerTemp, "manifest.json")}}]) {
      assert.throws(() => assertLinuxJitResourceResult(changed, manifest, manifestFile, manifestSha256), /does not bind/);
    }
    assert.throws(() => assertLinuxJitResourceResult(result, {...manifest, publicationQualified: true}, manifestFile, manifestSha256), /does not bind/);
    assert.throws(() => assertLinuxJitResourceResult(result, {...manifest, files: {}}, manifestFile, manifestSha256), /does not bind/);
    fs.writeFileSync(resultFile, JSON.stringify({...result, executed: true}) + "\n");
    assert.throws(() => sealLinuxJitResources(paths), /does not bind/, "a changed receipt is rejected before any output");
    fs.writeFileSync(resultFile, JSON.stringify(result) + "\n");
    const alias = path.join(runnerTemp, "alias.tar"); fs.linkSync(archive, alias);
    assert.throws(() => sealLinuxJitResources(paths), /canonical singly-linked/);
    fs.unlinkSync(alias); fs.unlinkSync(archive);
    fs.mkdirSync(path.join(base, "mom-native-linux-jit-resources-123456-1"));
    packState(archive, base, ["mom-native-linux-jit-resources-123456-1"]);
    assert.throws(() => sealLinuxJitResources(paths), /unexpected state roots/, "a stale run-attempt root cannot be sealed");
  });

  test("Windows JIT resource production requires every upstream output pin", () => {
    const {windowsJitInputPins} = require("../scripts/ci/linux-sdk");
    const names = ["MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256", "MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256",
        "MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256", "MOM_NATIVE_LINUX_JIT_RESOURCES_BUNDLE_SHA256",
        "MOM_NATIVE_LINUX_JIT_RESOURCES_RESULT_SHA256", "MOM_NATIVE_LINUX_JIT_RESOURCES_MANIFEST_SHA256",
        "MOM_NATIVE_WINDOWS_GNU_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_GNU_RESULT_SHA256", "MOM_NATIVE_WINDOWS_GNU_COPYRIGHTS_SHA256"],
      environment = Object.fromEntries(names.map((name, index) => [name, index.toString(16).repeat(64)]));
    assert.deepEqual(windowsJitInputPins(environment), environment);
    for (const name of names) {
      assert.throws(() => windowsJitInputPins({...environment, [name]: ""}), /output pins are required/, name);
    }
  });
  test("shared ROCm/COMGR core checks fresh outputs and LLVM admission before mutation", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-comgr-core-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = process.pid + "-fixture", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp),
        source: "/tmp/mom-native-prerequisites-" + suffix, sourceName: "mom-native-prerequisites-" + suffix,
        ioSourceName: "mom-native-io-prerequisites-" + suffix,
        sdkRoot: "/tmp/mom-native-llvm21-sdk-" + suffix, sdkName: "mom-native-llvm21-sdk-" + suffix,
        output: "/tmp/mom-native-llvm21-sdk-" + suffix + "/linux-llvm21-sdk"},
      environment = {MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
        MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix},
      state = rocmComgrStatePaths(paths, environment), tools = {cc: "/tools/icx", cxx: "/tools/icpx", cmake: "/tools/cmake"};
    let checks = 0;
    assert.throws(() => buildRocmComgrFromInputs(state, tools, {}, () => {checks++; throw new Error("fixture LLVM admission failed");}),
      /fixture LLVM admission failed/);
    assert.equal(checks, 1);
    for (const output of [state.resourcesRoot, state.hipHeadersRoot, state.sdkRoot]) {
      assert.equal(fs.existsSync(output), false, "preflight failure leaves output untouched: " + path.basename(output));
    }
    fs.writeFileSync(state.resourcesConfig, "preserve fixture\n");
    assert.throws(() => buildRocmComgrFromInputs(state, tools, {}, () => {checks++;}), /fresh canonical output\/config paths/);
    assert.equal(checks, 1, "an occupied config path is rejected before the LLVM callback");
    assert.equal(fs.readFileSync(state.resourcesConfig, "utf8"), "preserve fixture\n");
  });

  test("static OpenMP stage reuses the same-run pinned LLVM source tar and preserves only a build-only SDK handoff", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-openmp-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123456-2", source: "/tmp/mom-native-prerequisites-123456-2", sourceName: "mom-native-prerequisites-123456-2",
        runnerTemp: fs.realpathSync(runnerTemp)},
      state = openmpStatePaths(paths, {MOM_NATIVE_OPENMP_SDK_DIR: "/tmp/mom-native-openmp-sdk-123456-2"});
    assert.equal(state.archive, "/tmp/mom-native-prerequisites-123456-2/llvm-project-21.1.8.src.tar.xz");
    assert.equal(state.sdkRoot, "/tmp/mom-native-openmp-sdk-123456-2");
    assert.equal(path.basename(state.sourceTar), "native-llvm21-state-123456-2-sources.tar");
    assert.equal(state.sourceTar, path.join(runnerTemp, "native-llvm21-state-123456-2-sources.tar"));
    assert.equal(state.incomingTar, path.join(runnerTemp, "incoming/native-llvm21-state-123456-2-sources.tar"));
    assert.equal(state.sdkTar, path.join(runnerTemp, "native-linux-openmp-sdk-123456-2.tar"));
    assert.equal(state.toolsFile, path.join(runnerTemp, "native-openmp-tools-123456-2.json"));
    assert.deepEqual(openmpBuildCommand(state), {script: "scripts/build-native-openmp-sdk.js",
      args: [state.archive, state.sdkRoot, state.toolsFile]});
    assert.throws(() => openmpStatePaths(paths, {MOM_NATIVE_OPENMP_SDK_DIR: "/tmp/ambient-openmp"}), /exactly match/);
  });

  test("ACPP and descriptor transport pins reject mismatches before extraction or archive parsing", {skip: linuxOnlySkip}, t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-dpcpp-consumer-pin-")), archive = path.join(base, "cuda.tar"),
      stop = new Error("input inspection reached");
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    fs.writeFileSync(archive, "transport fixture\n");
    const state = {sdkRoot: "unused-sdk", sdkTar: "unused-output", incomingCudaInputsTar: archive,
      incoming: base, dpcpp: {sdkTar: archive}, root: "unused-root", archive: "unused-archive", acpp: {sourceName: "source"}};
    for (const entry of [buildAcppCudaHipSdk, buildLinuxNativeSdkDescriptor]) {
      let inspections = 0;
      const inspect = () => {inspections++; throw stop;},
        run = vm.runInNewContext("(" + entry.toString() + ")", {fs: {existsSync: () => false, lstatSync: () => undefined}, path,
          temp: base, canonicalExecutable: (/** @type {string} */ file) => file,
          acppCudaHipSdkStatePaths: () => state, linuxNativeSdkDescriptorStatePaths: () => state,
          dpcppCudaInputPins, dpcppSourceArchivePin, extractState: inspect, auditLinuxNativeSdkDescriptorInputArchives: inspect});
      const env = {MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: "0".repeat(64),
        MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: "1".repeat(64)};
      assert.throws(() => run({env}), /externally pinned same-run source tar/, entry.name);
      assert.equal(inspections, 0, entry.name + " rejects the bad transport before reading any input archive");
      assert.throws(() => run({env: {MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: fileHash(archive)}}), /both have external/);
      assert.equal(inspections, 0, "partial pins cannot fall back to legacy admission");
      assert.throws(() => run({env: {...env, MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: fileHash(archive)}}), error => error === stop);
      assert.equal(inspections, 1, "a verified transport proceeds to normal input admission");
      assert.throws(() => run({env: {}}), /both have external/);
      assert.equal(inspections, 1, "absent pins are rejected before any input is read");
    }
  });

  test("same-run ACPP CUDA/HIP stage consumes admitted same-run DPC++/CUDA and ROCm inputs", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-acpp-cuda-hip-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp),
      source: "/tmp/mom-native-prerequisites-" + suffix, sourceName: "mom-native-prerequisites-" + suffix,
      ioSourceName: "mom-native-io-prerequisites-" + suffix,
      sdkRoot: "/tmp/mom-native-llvm21-sdk-" + suffix, sdkName: "mom-native-llvm21-sdk-" + suffix,
      output: "/tmp/mom-native-llvm21-sdk-" + suffix + "/linux-llvm21-sdk"};
    const environment = {
      MOM_NATIVE_ACPP_SOURCE_DIR: "/tmp/mom-native-acpp-prerequisites-" + suffix,
      MOM_NATIVE_OPENMP_SDK_DIR: "/tmp/mom-native-openmp-sdk-" + suffix,
      MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
      MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
      MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
      MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix,
      MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/mom-native-rocm-host-sdk-" + suffix,
      MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
      MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix,
      MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR: "/tmp/mom-native-acpp-cuda-hip-sdk-" + suffix,
    };
    const state = acppCudaHipSdkStatePaths(paths, environment);
    assert.equal(state.acpp.preparedCheckout, environment.MOM_NATIVE_ACPP_SOURCE_DIR + "/prepared/acpp");
    assert.equal(state.llvmSdk, paths.output + "/sdk");
    assert.equal(state.rocmSdk, environment.MOM_NATIVE_ROCM_HOST_SDK_DIR + "/runtime/sdk");
    assert.equal(state.openmpLibrary, environment.MOM_NATIVE_OPENMP_SDK_DIR + "/sdk/libomp.a");
    assert.equal(state.incomingOpenmpTar, path.join(runnerTemp, "incoming", path.basename(state.openmp.sdkTar)));
    assert.notEqual(state.incomingOpenmpTar, state.openmp.incomingTar, "the ACPP consumer restores the OpenMP SDK artifact, not the LLVM source archive");
    assert.equal(state.cudaInputs.cudaRoot, environment.MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR + "/cuda");
    assert.equal(state.sdkTar, path.join(runnerTemp, "native-linux-acpp-cuda-hip-sdk-" + suffix + ".tar"));
    assert.throws(() => acppCudaHipSdkStatePaths(paths, {...environment, MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR: "/tmp/ambient-acpp"}), /exactly match/);

    const tools = {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"},
      command = acppCudaHipSdkBuildCommand(state, tools), env = {MOM_AOT_SYCL_BACKEND: "cuda-hip", MOM_AOT_ROCM_STATIC_ROOT: state.rocmSdk,
        MOM_AOT_LLD_ROOT: state.llvmSdk, ROCM_DEVICE_LIBS_PATH: state.deviceBitcode, MOM_AOT_COMGR_RECEIPT: state.comgrReceipt,
        MOM_AOT_OPENMP_LIBRARY: state.openmpLibrary},
      plan = staticSyclSdkPlan(command.args[0] || "", command.args[1] || "", command.args[2] || "", command.args[3] || "", 2, env, tools);
    assert.deepEqual(command, {script: "scripts/build-native-sycl-sdk.js", args: [state.acpp.preparedCheckout, state.llvmSdk,
      state.cudaInputs.cudaRoot, state.sdkRoot, tools.cc, tools.cxx]});
    assert.equal(plan.backend, "cuda-hip"); assert.equal(plan.usesCuda, true);
    assert.match(plan.cache, /CUDA_cudart_static_LIBRARY:FILEPATH=\/tmp\/mom-native-dpcpp-cuda-inputs-123456-2\/cuda\/lib64\/libcudart_static\.a/);
    assert.match(plan.cache, /CUDA_culibos_LIBRARY:FILEPATH=\/tmp\/mom-native-dpcpp-cuda-inputs-123456-2\/cuda\/lib64\/libculibos\.a/);
    assert.ok(plan.commands[0]?.includes("-DWITH_CUDA_BACKEND=ON"));
    assert.ok(plan.commands[0]?.includes("-DWITH_ROCM_BACKEND=ON"));
    assert.ok([...(plan.commands[1] || [])].includes("rt-backend-cuda"));
    assert.ok([...(plan.commands[1] || [])].includes("llvm-to-ptx"));
    assert.ok([...(plan.commands[1] || [])].includes("rt-backend-hip"));
    assert.ok([...(plan.commands[1] || [])].includes("llvm-to-amdgpu"));
    assert.deepEqual(plan.commands[1]?.slice(0, 5), ["cmake", "--build", plan.build, "--parallel", "2"]);
    assert.ok(plan.commands[0]?.includes("-DLLVM_DIR=" + state.llvmSdk + "/lib/cmake/llvm"));

    const hash = "a".repeat(64), deviceCxx = path.join(state.llvmSdk, "bin/clang++"),
      cudaHeaders = {"cuda.h": "c".repeat(64), "cuda_runtime_api.h": "d".repeat(64)},
      cudaInputs = {cuda: {root: state.cudaInputs.cudaRoot, headers: {files: cudaHeaders}, staticRuntime: {
        cudart: {path: path.join(state.cudaInputs.cudaRoot, "lib64/libcudart_static.a"), sha256: hash},
        culibos: {path: path.join(state.cudaInputs.cudaRoot, "lib64/libculibos.a"), sha256: "b".repeat(64)}}}},
      sdk = {libraries: [], backend: "cuda-hip", cudaVersion: 12060, source: state.acpp.preparedCheckout, build: path.join(state.sdkRoot, "build"), cxx: deviceCxx,
        comgr: {receipt: state.comgrReceipt, roots: {llvm: state.llvmSdk, lld: state.llvmSdk, device: state.deviceRoot}},
        openmp: {archive: state.openmpLibrary}, files: {[cudaInputs.cuda.staticRuntime.cudart.path]: hash,
          [cudaInputs.cuda.staticRuntime.culibos.path]: "b".repeat(64)}},
      record = {hostCompilers: tools, inputs: {comgr: {receipt: state.comgrReceipt}}, fingerprints: {
        [path.join(state.cudaInputs.cudaRoot, "include/cuda.h")]: cudaHeaders["cuda.h"],
        [path.join(state.cudaInputs.cudaRoot, "include/cuda_runtime_api.h")]: cudaHeaders["cuda_runtime_api.h"],
        [cudaInputs.cuda.staticRuntime.cudart.path]: hash, [cudaInputs.cuda.staticRuntime.culibos.path]: "b".repeat(64)}};
    assert.equal(assertAcppCudaHipSdk(state, record, sdk, tools, cudaInputs), sdk);
    assert.throws(() => assertAcppCudaHipSdk(state, record, {...sdk, cxx: tools.cxx}, tools, cudaInputs),
      /exact same-run CI inputs/, "device compilation must use this run's LLVM clang++, not the host compiler");
    assert.throws(() => assertAcppCudaHipSdk(state, {...record, hostCompilers: {...tools, cxx: deviceCxx}}, sdk, tools, cudaInputs),
      /exact same-run CI inputs/, "the separately recorded host compiler pair remains pinned to oneAPI");
    assert.throws(() => assertAcppCudaHipSdk(state, record, {...sdk, backend: "hip"}, tools, cudaInputs),
      /exact same-run CI inputs/);
    assert.throws(() => assertAcppCudaHipSdk(state, record, sdk, tools,
      {...cudaInputs, cuda: {...cudaInputs.cuda, root: "/tmp/foreign-cuda"}}), /exact same-run CI inputs/);
    assert.throws(() => assertAcppCudaHipSdk(state, {...record, fingerprints: {...record.fingerprints,
      [path.join(state.cudaInputs.cudaRoot, "include/cuda.h")]: "e".repeat(64)}}, sdk, tools, cudaInputs), /exact same-run CI inputs/);
    const withoutHeaders = {...record, fingerprints: {
      [cudaInputs.cuda.staticRuntime.cudart.path]: hash, [cudaInputs.cuda.staticRuntime.culibos.path]: "b".repeat(64)}};
    assert.throws(() => assertAcppCudaHipSdk(state, withoutHeaders, sdk, tools,
      {...cudaInputs, cuda: {...cudaInputs.cuda, headers: {files: {}}}}), /exact same-run CI inputs/);
    assert.throws(() => assertAcppCudaHipSdk(state, record, {...sdk, files: {...sdk.files, [cudaInputs.cuda.staticRuntime.cudart.path]: "0".repeat(64)}}, tools, cudaInputs),
      /exact same-run CI inputs/);
  });

});

test.describe("Native prerequisite provenance and transport", () => {
  test("Hermes admission binds the exact source pin, current fetch recipe, patch and producer", () => {
    assert.equal(hermesPins.revision, "26fbb82fad39d9d06901adb1d6d630dacc84c877");
    assert.equal(hermesPins.promiseSha256, "071f4322f9dba4fa06992065bc5f1eb9b11e49c9b1b6336d31f377b3e9801184");
    const recipe = sourceRecipe("hermes"), source = "/tmp/mom-native-hermes-prerequisites-123456-2",
      git = "/usr/bin/git", gitSha256 = "a".repeat(64), fetcherSha256 = "b".repeat(64), root = path.join(source, "roots/hermes"),
      receipt = {version: 1, target: "hermes", status: "PASS", networked: true, buildOnly: true, compiled: false,
        publicationQualified: false, git: {path: git, sha256: gitSha256}, fetcherSha256,
        recipe: {sha256: createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), files: recipe.graph},
        components: {hermes: {origin: "https://github.com/facebook/hermes.git", revision: hermesPins.revision, root,
          gitDirectory: "bare", shallow: true, partialClone: false}}}, roots = {hermes: root}, original = structuredClone(receipt);
    assert.ok(Object.hasOwn(recipe.graph, "scripts/build-native-hermes.js"));
    assert.ok(Object.hasOwn(recipe.graph, "runtime/hermes-host-rejections.patch"));
    assert.doesNotThrow(() => assertHermesFetchProvenance(receipt, roots, source, git, gitSha256, fetcherSha256));
    assert.deepEqual(receipt, original, "admission does not restamp or mutate the fetch receipt");
    for (const altered of [
      {...structuredClone(receipt), components: {hermes: {...receipt.components.hermes, revision: "0".repeat(40)}}},
      {...structuredClone(receipt), recipe: {...receipt.recipe, files: {...receipt.recipe.files, "scripts/build-native-hermes.js": "0".repeat(64)}}},
      {...structuredClone(receipt), git: {...receipt.git, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), publicationQualified: true},
    ]) {assert.throws(() => assertHermesFetchProvenance(altered, roots, source, git, gitSha256, fetcherSha256), /does not bind/);}
    assert.throws(() => assertHermesFetchProvenance(receipt, {hermes: "/tmp/unrelated-root"}, source, git, gitSha256, fetcherSha256), /does not bind/);
  });

  test("ACPP source admission binds the shared pin, fetch recipe and original bare root", () => {
    const recipe = sourceRecipe("acpp"), source = "/tmp/mom-native-acpp-prerequisites-123456-2",
      git = "/usr/bin/git", gitSha256 = "a".repeat(64), fetcherSha256 = "b".repeat(64), root = path.join(source, "roots/acpp"),
      receipt = {version: 1, target: "acpp", status: "PASS", networked: true, buildOnly: true, compiled: false,
        publicationQualified: false, git: {path: git, sha256: gitSha256}, fetcherSha256,
        recipe: {sha256: createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), files: recipe.graph},
        components: {acpp: {origin: origins.acpp, revision: syclRevision, root, gitDirectory: "bare", shallow: true, partialClone: false}}},
      roots = {acpp: root}, original = structuredClone(receipt);
    assert.ok(Object.hasOwn(recipe.graph, "scripts/build-native-sycl-sdk.js"));
    assert.ok(Object.hasOwn(recipe.graph, "runtime/adaptivecpp-static.patch"));
    assert.ok(Object.hasOwn(recipe.graph, "runtime/adaptivecpp-jit.patch"));
    assert.doesNotThrow(() => assertAcppFetchProvenance(receipt, roots, source, git, gitSha256, fetcherSha256));
    assert.deepEqual(receipt, original, "admission does not restamp or mutate the fetch receipt");
    for (const altered of [
      {...structuredClone(receipt), target: "hermes"},
      {...structuredClone(receipt), networked: false},
      {...structuredClone(receipt), compiled: true},
      {...structuredClone(receipt), publicationQualified: true},
      {...structuredClone(receipt), git: {...receipt.git, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), fetcherSha256: "0".repeat(64)},
      {...structuredClone(receipt), recipe: {...receipt.recipe, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), recipe: {...receipt.recipe, files: {...receipt.recipe.files, "scripts/build-native-sycl-sdk.js": "0".repeat(64)}}},
      {...structuredClone(receipt), components: {acpp: {...receipt.components.acpp, origin: "https://example.invalid/repo.git"}}},
      {...structuredClone(receipt), components: {acpp: {...receipt.components.acpp, revision: "0".repeat(40)}}},
      {...structuredClone(receipt), components: {acpp: {...receipt.components.acpp, root: "/tmp/unrelated-root"}}},
      {...structuredClone(receipt), components: {acpp: {...receipt.components.acpp, partialClone: true}}},
      {...structuredClone(receipt), components: {acpp: {...receipt.components.acpp, extra: true}}},
    ]) {assert.throws(() => assertAcppFetchProvenance(altered, roots, source, git, gitSha256, fetcherSha256), /does not bind/);}
    assert.throws(() => assertAcppFetchProvenance(receipt, {acpp: "/tmp/unrelated-root"}, source, git, gitSha256, fetcherSha256), /does not bind/);
    assert.throws(() => assertAcppFetchProvenance(receipt, {acpp: root, extra: root}, source, git, gitSha256, fetcherSha256), /does not bind/);
  });

  test("ACPP prepared-source bridge binds the exact run path and shared revision", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-acpp-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123456-2", runnerTemp: fs.realpathSync(runnerTemp)},
      environment = {MOM_NATIVE_ACPP_SOURCE_DIR: "/tmp/mom-native-acpp-prerequisites-123456-2"},
      state = acppStatePaths(paths, environment), receipt = {target: "acpp", prepared: {acpp: {root: state.preparedCheckout, revision: syclRevision}}};
    assert.equal(state.sourceName, "mom-native-acpp-prerequisites-123456-2");
    assert.equal(state.prepared, path.join(state.source, "prepared"));
    assert.equal(state.preparedCheckout, path.join(state.prepared, "acpp"));
    assert.equal(path.dirname(state.source), "/tmp");
    assert.equal(path.basename(state.sourceTar), "native-acpp-state-123456-2-sources.tar");
    assert.equal(path.basename(state.incomingTar), "native-acpp-state-123456-2-sources.tar");
    assert.doesNotThrow(() => assertAcppPreparedSource(receipt, state));
    for (const altered of [
      {...receipt, target: "hermes"},
      {target: "acpp", prepared: {acpp: {...receipt.prepared.acpp, root: "/tmp/ambient-acpp"}}},
      {target: "acpp", prepared: {acpp: {...receipt.prepared.acpp, revision: "0".repeat(40)}}},
    ]) {assert.throws(() => assertAcppPreparedSource(altered, state), /does not bind/);}
    assert.throws(() => acppStatePaths(paths, {...environment, MOM_NATIVE_ACPP_SOURCE_DIR: "/tmp/ambient-acpp"}), /exactly match/);
  });

  test("Linux source handoff fetches one compiler cohort and binds both prepared target views", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-linux-source-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const source = "/tmp/mom-native-linux-source-prerequisites-123456-2", git = "/usr/bin/git",
      gitSha256 = "a".repeat(64), fetcherSha256 = "b".repeat(64), recipe = sourceRecipe("compiler"), selected = sdkComponents("compiler"),
      names = /** @type {(keyof typeof selected)[]} */ (Object.keys(selected).sort()), linux = sdkComponents("linux"),
      linuxNames = /** @type {(keyof typeof linux)[]} */ (Object.keys(linux).sort()),
      roots = Object.fromEntries(names.map(name => [name, path.join(source, "roots", name)]));
    const components = Object.fromEntries(names.map(name => {
      const component = selected[name]; assert.ok(component);
      return [name, {origin: origins[name], revision: component.revision, root: roots[name], gitDirectory: "bare", shallow: true, partialClone: false}];
    }));
    const receipt = {version: 1, target: "compiler", status: "PASS", networked: true, buildOnly: true, compiled: false,
      publicationQualified: false, git: {path: git, sha256: gitSha256}, fetcherSha256,
      recipe: {sha256: createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), files: recipe.graph}, components};
    const original = structuredClone(receipt), paths = {suffix: "123456-2", runnerTemp: fs.realpathSync(runnerTemp)},
      state = linuxSourceStatePaths(paths, {MOM_NATIVE_LINUX_SOURCES_DIR: source});
    const preparedLinuxReceipt = {version: 2, target: "linux", status: "PASS", compiled: false, buildOnly: true, publicationQualified: false,
      prepared: Object.fromEntries(linuxNames.map(name => {
        const component = linux[name]; assert.ok(component);
        return [name, {root: path.join(state.linuxPrepared, name), revision: component.revision}];
      }))};
    const preparedCompilerReceipt = {version: 2, target: "compiler", status: "PASS", compiled: false, buildOnly: true, publicationQualified: false,
      prepared: Object.fromEntries(names.map(name => {
        const component = selected[name]; assert.ok(component);
        return [name, {root: path.join(state.compilerPrepared, name), revision: component.revision}];
      }))};
    assert.ok(Object.hasOwn(recipe.graph, "scripts/prepare-native-sdk-sources.js"));
    assert.ok(Object.hasOwn(recipe.graph, "scripts/native-dpcpp-pins.js"));
    assert.equal(names.length, 10);
    assert.equal(linuxNames.length, 8);
    assert.equal(dpcppPins.source, selected.dpcpp?.revision);
    assert.equal(state.sourceName, "mom-native-linux-source-prerequisites-123456-2");
    assert.equal(state.prepared, path.join(source, "prepared"));
    assert.equal(state.linuxPrepared, path.join(state.prepared, "linux"));
    assert.equal(state.compilerPrepared, path.join(state.prepared, "compiler"));
    assert.equal(state.rootsFile, path.join(source, "roots.json"));
    assert.equal(state.fetchReceipt, path.join(source, "fetch.json"));
    assert.equal(state.preparedReceipt, path.join(state.linuxPrepared, "sources.json"));
    assert.equal(state.compilerPreparedReceipt, path.join(state.compilerPrepared, "sources.json"));
    assert.equal(path.basename(state.sourceTar), "native-linux-sdk-state-123456-2-sources.tar");
    assert.equal(path.basename(state.incomingTar), "native-linux-sdk-state-123456-2-sources.tar");
    assert.equal(path.dirname(state.source), "/tmp");
    assert.deepEqual(linuxSourceFetchArgs(state, git), ["compiler", git, state.source, "--allow-network"],
      "the shared source handoff performs exactly one fetch against the 10-root compiler target");
    const plan = linuxSourcePreparationPlan(state, roots);
    const linuxPlan = plan[0], compilerPlan = plan[1];
    assert.ok(linuxPlan && compilerPlan);
    assert.deepEqual(plan, [
      {target: "linux", roots: Object.fromEntries(linuxNames.map(name => [name, roots[name]])), output: state.linuxPrepared},
      {target: "compiler", roots: Object.fromEntries(names.map(name => [name, roots[name]])), output: state.compilerPrepared},
    ]);
    assert.equal(Object.keys(linuxPlan.roots).length, 8);
    assert.equal(Object.keys(compilerPlan.roots).length, 10);
    assert.doesNotThrow(() => assertSdkFetchProvenance(receipt, roots, "compiler", source, fetcherSha256, {path: git, sha256: gitSha256}));
    const historicalFetcher = {...structuredClone(receipt), fetcherSha256: "c".repeat(64)};
    assert.doesNotThrow(() => assertSdkFetchProvenance(historicalFetcher, roots, "compiler", source),
      "offline consumers accept a well-formed historical fetcher digest while checking the current source recipe");
    assert.throws(() => assertSdkFetchProvenance(historicalFetcher, roots, "compiler", source, fetcherSha256, {path: git, sha256: gitSha256}),
      /does not bind/, "fresh acquisition must bind the receipt to the fetcher that just ran");
    assert.throws(() => assertSdkFetchProvenance({...receipt, fetcherSha256: "malformed"}, roots, "compiler", source),
      /does not bind/, "historical fetcher identity remains strict SHA-256 metadata");
    const differentProducer = {...structuredClone(receipt), git: {path: "/opt/acquisition/git", sha256: "c".repeat(64)}},
      differentOriginal = structuredClone(differentProducer);
    assert.doesNotThrow(() => assertSdkFetchProvenance(differentProducer, roots, "compiler", source),
      "consumer verifies the source cohort without requiring the original acquisition executable");
    assert.deepEqual(differentProducer, differentOriginal, "consumer never restamps original Git identity");
    assert.throws(() => assertSdkFetchProvenance(differentProducer, roots, "compiler", source, fetcherSha256,
      {path: git, sha256: gitSha256}), /does not bind/, "acquisition must still bind its actual Git identity");
    for (const git of [{path: "relative/git", sha256: "c".repeat(64)}, {path: "/usr/bin/git", sha256: "malformed"},
      {path: "/usr/bin/git", sha256: ["c".repeat(64)]}]) {
      assert.throws(() => assertSdkFetchProvenance({...differentProducer, git}, roots, "compiler", source, fetcherSha256), /does not bind/);
    }
    assert.deepEqual(receipt, original, "admission does not mutate the immutable pinned fetch receipt");
    assert.doesNotThrow(() => assertPreparedSdkPaths(preparedLinuxReceipt, "linux", state.linuxPrepared));
    assert.doesNotThrow(() => assertPreparedSdkPaths(preparedCompilerReceipt, "compiler", state.compilerPrepared));
    for (const altered of [
      {...structuredClone(receipt), target: "linux"},
      {...structuredClone(receipt), networked: false},
      {...structuredClone(receipt), publicationQualified: true},
      {...structuredClone(receipt), git: {...receipt.git, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), recipe: {...receipt.recipe, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), components: {...receipt.components, dpcpp: {...receipt.components["dpcpp"], revision: "0".repeat(40)}}},
      {...structuredClone(receipt), components: {...receipt.components, dpcpp: {...receipt.components["dpcpp"], origin: "https://example.invalid/repo.git"}}},
      {...structuredClone(receipt), components: {...receipt.components, vcIntrinsics: {...receipt.components["vcIntrinsics"], revision: "0".repeat(40)}}},
      {...structuredClone(receipt), components: Object.fromEntries(Object.entries(receipt.components).filter(([name]) => !["vcIntrinsics", "spirvHeaders"].includes(name)))},
    ]) {assert.throws(() => assertSdkFetchProvenance(altered, roots, "compiler", source, fetcherSha256, {path: git, sha256: gitSha256}), /does not bind|changed its pinned/);}
    assert.throws(() => assertSdkFetchProvenance(receipt, {...roots, extra: source}, "compiler", source, fetcherSha256), /does not bind/);
    assert.throws(() => assertSdkFetchProvenance(receipt, {...roots, emhash: "/tmp/drifted-emhash"}, "compiler", source, fetcherSha256), /changed its pinned/);
    assert.throws(() => linuxSourcePreparationPlan(state, Object.fromEntries(Object.entries(roots).filter(([name]) => name !== "spirvHeaders"))), /exact ten/);
    assert.throws(() => linuxSourcePreparationPlan(state, {...roots, unexpected: source}), /exact ten/);
    assert.throws(() => linuxSourcePreparationPlan(state, {...roots, dpcpp: "/tmp/drifted-dpcpp"}), /exact ten/);
    assert.throws(() => linuxSourceStatePaths(paths, {MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/ambient-linux-sources"}), /exactly match/);
    for (const altered of [
      {...preparedLinuxReceipt, target: "compiler"},
      {...preparedLinuxReceipt, prepared: {...preparedLinuxReceipt.prepared, dpcpp: {...preparedLinuxReceipt.prepared["dpcpp"], root: "/tmp/ambient"}}},
      {...preparedLinuxReceipt, prepared: {...preparedLinuxReceipt.prepared, hwloc: {...preparedLinuxReceipt.prepared["hwloc"], revision: "0".repeat(40)}}},
    ]) {assert.throws(() => assertPreparedSdkPaths(altered, "linux", state.linuxPrepared), /does not bind|changed its run-scoped/);}
    for (const altered of [
      {...preparedCompilerReceipt, target: "linux"},
      {...preparedCompilerReceipt, prepared: Object.fromEntries(Object.entries(preparedCompilerReceipt.prepared).filter(([name]) => name !== "vcIntrinsics"))},
      {...preparedCompilerReceipt, prepared: {...preparedCompilerReceipt.prepared, spirvHeaders: {...preparedCompilerReceipt.prepared["spirvHeaders"], root: "/tmp/ambient"}}},
    ]) {assert.throws(() => assertPreparedSdkPaths(altered, "compiler", state.compilerPrepared), /does not bind|changed its run-scoped/);}
  });

  test("real CMake FindCUDAToolkit accepts only a minimal synthetic CUDA root", {skip: process.platform !== "linux" || !fs.existsSync("/usr/bin/cmake")}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cmake-cuda-probe-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = process.pid + "-cmake-fixture", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)},
      environment = {MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix},
      state = dpcppCudaInputsStatePaths(paths, environment);
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    fs.mkdirSync(path.join(state.cudaRoot, "bin"), {recursive: true});
    for (const header of ["cuda.h", "cuda_runtime.h", "nvml.h"]) {
      const file = path.join(state.cudaRoot, "include", header); fs.mkdirSync(path.dirname(file), {recursive: true});
      fs.writeFileSync(file, "/* configure-only fixture */\n");
    }
    fs.mkdirSync(path.join(state.cudaRoot, "lib64"), {recursive: true});
    fs.writeFileSync(path.join(state.cudaRoot, "lib64/libcudart.so"), "synthetic link input\n");
    const nvcc = path.join(state.cudaRoot, "bin/nvcc"), nvccScript = "#!/bin/sh\n" +
      "if [ \"$1\" = \"--version\" ]; then\n  printf '%s\\n' 'Cuda compilation tools, release 12.6, V12.6.85'\n" +
      "elif [ \"$1\" = \"-v\" ] && [ \"$2\" = \"__cmake_determine_cuda\" ]; then\n  printf '#$ TOP=%s\\n' '" + state.cudaRoot + "'\n" +
      "else exit 2; fi\n";
    fs.writeFileSync(nvcc, nvccScript, {mode: 0o755}); fs.chmodSync(nvcc, 0o755);
    const evidence = probeCudaToolkit(state, {env: {PATH: "/usr/bin:/bin", HOME: runnerTemp, LC_ALL: "C"}});
    assert.ok(typeof evidence.result.toolkitVersion === "string");
    assert.match(evidence.result.toolkitVersion, /^12\.6\./);
    assert.equal(evidence.result.cudart, path.join(state.cudaRoot, "lib64/libcudart.so"));
    assert.equal(evidence.result.importedCudart, evidence.result.cudart);
    assert.ok(typeof evidence.result.includeDirectories === "string");
    assert.ok(evidence.result.includeDirectories.split(";").includes(path.join(state.cudaRoot, "include")));
    assert.equal(fs.existsSync(state.cudaCmakeBuild), false, "the configure-only smoke leaves no reusable build tree");
  });

  test("NVCC toolkit metadata accepts only its expected sentinel failure at the selected root", () => {
    const root = path.resolve(os.tmpdir(), "mom-native-ci-cuda-toolkit"), version = {status: 0, signal: null,
        stdout: "Cuda compilation tools, release 12.6, V12.6.85\n", stderr: ""},
      probe = {status: 1, signal: null, stdout: "", stderr: "#$ TOP=" + path.join(root, "bin", "..") +
        "\nnvcc fatal   : Don't know what to do with '__cmake_determine_cuda'\n"};
    assert.equal(cudaToolkitProbeRoot(version, probe, root), root);
    assert.equal(cudaToolkitProbeRoot(version, {...probe, status: 0}, root), root);
    assert.throws(() => cudaToolkitProbeRoot({...version, status: 1}, probe, root), /toolkit probe/);
    assert.throws(() => cudaToolkitProbeRoot(version, {...probe, status: 2}, root), /toolkit probe/);
    assert.throws(() => cudaToolkitProbeRoot(version, {...probe, signal: "SIGTERM"}, root), /toolkit probe/);
    assert.throws(() => cudaToolkitProbeRoot(version, {...probe, stderr: "#$ TOP=/new/cuda\nreal compiler failure\n"}, root), /toolkit probe/);
    assert.throws(() => cudaToolkitProbeRoot(version, {...probe,
      stderr: probe.stderr.replace(path.join(root, "bin", ".."), path.join(root, "ambient"))}, root), /toolkit probe/);
    assert.throws(() => cudaToolkitProbeRoot(version, {...probe, stderr: ""}, root), /toolkit probe/);
  });

  test("DPC++ staging preserves relative compiler aliases inside the copied SDK", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-dpcpp-copy-")),
      source = path.join(root, "source"), output = path.join(root, "output");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.mkdirSync(path.join(source, "bin"), {recursive: true});
    fs.writeFileSync(path.join(source, "bin/clang-23"), "pinned compiler fixture\n", {mode: 0o755});
    fs.symlinkSync("clang-23", path.join(source, "bin/clang"));
    fs.symlinkSync("clang", path.join(source, "bin/clang++"));
    copyDpcppBuildInputs(source, output);
    assert.equal(fs.readlinkSync(path.join(output, "bin/clang++")), "clang");
    assert.equal(fs.readlinkSync(path.join(output, "bin/clang")), "clang-23");
    assert.equal(fs.realpathSync(path.join(output, "bin/clang++")), path.join(output, "bin/clang-23"));
    assert.equal(fileHash(path.join(output, "bin/clang++")), fileHash(path.join(source, "bin/clang-23")));
    assert.throws(() => copyDpcppBuildInputs(source, output), /fresh output/);
    assert.throws(() => copyDpcppBuildInputs(source, path.join(source, "nested")), /disjoint roots/);
    const dangling = path.join(root, "dangling"); fs.symlinkSync("missing", dangling);
    assert.throws(() => copyDpcppBuildInputs(source, dangling), /fresh output/);
  });

  test("DPC++ input source tar requires an external pin before any extraction", {skip: linuxOnlySkip}, t => {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-dpcpp-source-pin-")), file = path.join(root, "source.tar");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(file, "source tar fixture\n"); const sha256 = fileHash(file);
    assert.equal(dpcppSourceArchivePin(file, sha256), sha256);
    for (const pin of ["", "0".repeat(64), "malformed"]) {
      assert.throws(() => dpcppSourceArchivePin(file, pin), /externally pinned/);
    }
    const alias = path.join(root, "alias"); fs.symlinkSync("source.tar", alias);
    assert.throws(() => dpcppSourceArchivePin(alias, sha256), /externally pinned/);
    const hardlink = path.join(root, "hardlink"); fs.linkSync(file, hardlink);
    assert.throws(() => dpcppSourceArchivePin(file, sha256), /externally pinned/);
    assert.equal(fs.readFileSync(file, "utf8"), "source tar fixture\n", "rejections leave inputs unchanged");
  });

  test("synthetic DPC++23/CUDA 12.6 handoff manifest is run-scoped and hash-bound", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-dpcpp-inputs-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = process.pid + "-fixture", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)},
      environment = {MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix},
      state = dpcppCudaInputsStatePaths(paths, environment);
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    assert.equal(state.linux.sourceName, "mom-native-linux-source-prerequisites-" + suffix);
    assert.equal(state.root, environment.MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR);
    assert.equal(path.basename(state.incomingSourceTar), "native-linux-sdk-state-" + suffix + "-sources.tar");
    assert.equal(path.basename(state.sdkTar), "native-linux-dpcpp-cuda-inputs-" + suffix + ".tar");
    assert.throws(() => dpcppCudaInputsStatePaths(paths, {...environment, MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/stale-inputs"}), /exactly match/);

    /** @param {string} base @param {string} relative @param {string|Buffer} content @param {number} [mode] */
    const put = (base, relative, content, mode = 0o644) => {
      const file = path.join(base, ...relative.split("/")); fs.mkdirSync(path.dirname(file), {recursive: true});
      fs.writeFileSync(file, content, {mode}); fs.chmodSync(file, mode); return file;
    };
    fs.mkdirSync(state.root);
    for (const file of [state.linux.fetchReceipt, state.linux.preparedReceipt, state.linux.compilerPreparedReceipt]) {
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture receipt\n");
    }
    put(state.dpcppRoot, ".mom-toolchain-sha256", dpcppPins.sdk + "\n");
    const version = `clang version 23.0.0git (https://github.com/intel/llvm ${dpcppPins.source})\n`;
    put(state.dpcppRoot, "bin/clang", "pinned clang\n", 0o755); put(state.dpcppRoot, "bin/clang++", "pinned clang++\n", 0o755);
    put(state.dpcppRoot, "lib/libsycl.so.9", "sycl\n"); put(state.dpcppRoot, "lib/libsycl-jit.so", "jit\n");
    put(state.cudaRoot, "bin/nvcc", "nvcc\n", 0o755); put(state.cudaRoot, "bin/ptxas", "ptxas\n", 0o755);
    put(state.cudaRoot, "bin/fatbinary", "fatbinary\n", 0o755);
    put(state.cudaRoot, "bin/nvcc.profile", "TOP = $(_HERE_)/..\n");
    put(state.cudaRoot, "include/cuda.h", "#define CUDA_VERSION 12060\n");
    put(state.cudaRoot, "include/nvml.h", "/* NVML interface */\n");
    put(state.cudaRoot, "include/cuda_runtime.h", "/* CUDA runtime headers */\n");
    put(state.cudaRoot, "include/cuda/std/cstdint", "/* CCCL */\n");
    put(state.cudaRoot, "nvvm/libdevice/libdevice.10.bc", "device bitcode\n");
    put(state.cudaRoot, "lib64/libcudart.so", "minimal CMake link input\n");
    put(state.cudaRoot, "lib64/libcudart_static.a", "!<arch>\n"); put(state.cudaRoot, "lib64/libculibos.a", "!<arch>\n");
    const packageVersions = {"cuda-cudart-12-6": "12.6.85-1", "cuda-cudart-dev-12-6": "12.6.85-1", "cuda-nvml-dev-12-6": "12.6.77-1"},
      cudaCopyrights = Object.entries(packageVersions).map(([name, version]) => {
        const file = put(state.root, "licenses/" + name + "/copyright", name + " package copyright\n");
        return {package: name, version, sourcePath: "/usr/share/doc/" + name + "/copyright", path: file, sha256: fileHash(file)};
      }), copyright = path.join(state.root, "licenses/cuda-cudart-12-6/copyright");
    const nvccVersion = "Cuda compilation tools, release 12.6, V12.6.85\n", probe = "#$ TOP=" + state.cudaRoot + "\n#$ INCLUDES= -I" + path.join(state.cudaRoot, "include") + "\n";
    fs.writeFileSync(state.cudaVersionEvidence, nvccVersion); fs.writeFileSync(state.cudaProbeEvidence, probe);
    const cmakeLists = cudaToolkitProbeCMakeLists(), cmake = "/usr/bin/cmake", cmakeSha256 = "a".repeat(64),
      command = [cmake, "-S", state.cudaCmakeSource, "-B", state.cudaCmakeBuild, "-DCUDAToolkit_ROOT=" + state.cudaRoot];
    fs.mkdirSync(state.cudaCmakeSource, {recursive: true}); fs.writeFileSync(state.cudaCmakeLists, cmakeLists);
    // This fixture tests manifest admission only; the producer performs a real configure-only smoke.
    const cmakeEvidence = {version: 1, command, status: 0, tool: {path: cmake, sha256: cmakeSha256},
      script: {path: state.cudaCmakeLists, sha256: createHash("sha256").update(cmakeLists).digest("hex")},
      result: {toolkitVersion: "12.6.85", cudart: path.join(state.cudaRoot, "lib64/libcudart.so"),
        importedCudart: path.join(state.cudaRoot, "lib64/libcudart.so"), includeDirectories: path.join(state.cudaRoot, "include")},
      stdout: "-- DPCPP_CUDA_TOOLKIT_PROBE=PASS\n", stderr: ""};
    fs.writeFileSync(state.cudaCmakeEvidence, JSON.stringify(cmakeEvidence, null, 2) + "\n");
    assert.match(cmakeLists, /find_package\(CUDAToolkit 10\.1 REQUIRED\)/);
    assert.match(cmakeLists, /TARGET CUDA::cudart/);
    assert.doesNotMatch(cmakeLists, /add_executable|add_library|try_compile/i);
    const record = dpcppCudaInputsManifest(state, "a".repeat(64), packageVersions,
      {version: nvccVersion, top: state.cudaRoot, dpcppVersion: version}, cudaCopyrights, []);
    assert.equal(record.producer.helperSha256, fileHash(require.resolve("../scripts/native-dpcpp-cuda-inputs")),
      "new records bind the narrow shared producer, not unrelated CI transport edits");
    assert.equal(auditDpcppCudaInputs(state, record), record);
    assert.deepEqual(Object.keys(record.source).sort(), ["compilerPreparationSha256", "fetchReceiptSha256", "linuxPreparationSha256", "name", "tar", "tarSha256"],
      "the existing CI tar-origin schema stays unchanged");
    assert.equal(record.source.tarSha256, "a".repeat(64));
    assert.equal(Object.hasOwn(record.source, "origin"), false);

    const localWorkspace = path.join(runnerTemp, "marked-workspace"); fs.mkdirSync(localWorkspace);
    fs.writeFileSync(path.join(localWorkspace, ".mom-native-sdk-workspace.json"), JSON.stringify({version: 1, checkout: root}) + "\n");
    const localLinux = {...state.linux,
        fetchReceipt: put(localWorkspace, "compiler-fetch/fetch.json", "pinned fetch receipt fixture\n"),
        preparedReceipt: put(localWorkspace, "linux-sources/sources.json", "prepared Linux receipt fixture\n"),
        compilerPreparedReceipt: put(localWorkspace, "compiler-sources/sources.json", "prepared compiler receipt fixture\n")},
      localState = {...state, workspaceRoot: localWorkspace, linux: localLinux},
      localOrigin = preparedDpcppCudaSourceOrigin(localLinux, localWorkspace),
      localRecord = dpcppCudaInputsManifest(localState, localOrigin, packageVersions,
        {version: nvccVersion, top: state.cudaRoot, dpcppVersion: version}, cudaCopyrights, []);
    assert.equal(localRecord.producer.helperSha256, fileHash(require.resolve("../scripts/native-dpcpp-cuda-inputs")));
    assert.equal(localRecord.source.origin, "workspace-prepared-receipts");
    assert.deepEqual(Object.keys(localRecord.source).sort(), ["compilerPreparation", "fetchReceipt", "linuxPreparation", "origin", "workspaceRoot"]);
    assert.equal(Object.hasOwn(localRecord.source, "tarSha256"), false, "local prepared sources never claim an archive hash");
    assert.equal(auditDpcppCudaInputs(localState, localRecord), localRecord);
    const staleSource = fs.readFileSync(localLinux.preparedReceipt);
    fs.writeFileSync(localLinux.preparedReceipt, "stale receipt\n");
    assert.throws(() => auditDpcppCudaInputs(localState, localRecord), /provenance is malformed or stale/);
    fs.writeFileSync(localLinux.preparedReceipt, staleSource);
    for (const source of [
      {...localRecord.source, fetchReceipt: {...localRecord.source.fetchReceipt, path: path.join(runnerTemp, "outside-fetch.json")}},
      {...localRecord.source, tarSha256: "a".repeat(64)},
      {...localRecord.source, origin: "unknown-origin"},
    ]) {
      assert.throws(() => auditDpcppCudaInputs(localState, {...localRecord, source}), /provenance is malformed or stale/);
    }
    const fetchBackup = localLinux.fetchReceipt + ".saved", outsideFetch = path.join(runnerTemp, "outside-fetch.json");
    fs.writeFileSync(outsideFetch, fs.readFileSync(localLinux.fetchReceipt)); fs.renameSync(localLinux.fetchReceipt, fetchBackup);
    try {
      fs.symlinkSync(outsideFetch, localLinux.fetchReceipt);
      assert.throws(() => auditDpcppCudaInputs(localState, localRecord), /provenance is malformed or stale/);
    } finally {fs.unlinkSync(localLinux.fetchReceipt); fs.renameSync(fetchBackup, localLinux.fetchReceipt);}
    const outsideState = {...localState, linux: {...localLinux, fetchReceipt: path.join(os.tmpdir(), "outside-fetch-receipt.json")}};
    assert.throws(() => auditDpcppCudaInputs(outsideState, localRecord), /provenance is malformed or stale/);
    const unmarked = path.join(runnerTemp, "unmarked-workspace"); fs.mkdirSync(unmarked);
    assert.throws(() => preparedDpcppCudaSourceOrigin(localLinux, unmarked), /canonical marked workspace root/);

    const bundleSha256 = "a".repeat(64), manifestSha256 = "b".repeat(64);
    assert.throws(() => dpcppCudaInputPins({}), /both have external SHA-256 pins/);
    assert.throws(() => dpcppCudaInputPins({MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: bundleSha256}), /both have external SHA-256 pins/);
    assert.throws(() => dpcppCudaInputPins({MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: manifestSha256}), /both have external SHA-256 pins/);
    assert.throws(() => dpcppCudaInputPins({MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: "bad",
      MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: manifestSha256}), /both have external SHA-256 pins/);
    assert.deepEqual(dpcppCudaInputPins({MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: bundleSha256,
      MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: manifestSha256}), {bundleSha256, manifestSha256});

    /** @param {unknown} value */
    const writeManifest = value => fs.writeFileSync(state.manifest, JSON.stringify(value, null, 2) + "\n");
    writeManifest(record);
    const sealedSha256 = fileHash(state.manifest);
    assert.equal(auditDpcppCudaInputs(state, record, sealedSha256), record);
    assert.throws(() => auditDpcppCudaInputs(state, record, "0".repeat(64)), /manifest differs from its external same-run pin/);
    const unsealedMutation = structuredClone(record); unsealedMutation.producer.helperSha256 = "c".repeat(64);
    assert.throws(() => auditDpcppCudaInputs(state, unsealedMutation, sealedSha256), /record does not match its externally pinned manifest/);
    const historical = structuredClone(record);
    historical.producer.helperSha256 = "c".repeat(64);
    historical.producer.installer.sha256 = "d".repeat(64);
    writeManifest(historical);
    const historicalSha256 = fileHash(state.manifest);
    assert.equal(auditDpcppCudaInputs(state, historical, historicalSha256), historical,
      "only an externally pinned unchanged manifest permits its historical acquisition helper/installer hashes");
    assert.throws(() => auditDpcppCudaInputs(state, historical), /provenance is malformed or stale/,
      "without external pins the current helper/installer equality remains strict");
    fs.appendFileSync(state.manifest, " ");
    assert.throws(() => auditDpcppCudaInputs(state, historical, historicalSha256), /manifest differs from its external same-run pin/);
    writeManifest(record);
    assert.equal(record.compiled, false); assert.equal(record.executed, false); assert.equal(record.publicationQualified, false);
    assert.match(record.scope, /static CUDA archives are carried for the source-built ACPP runtime, which is built separately/);
    assert.deepEqual(Object.keys(record.cuda.packages).sort(), ["cuda-cudart-12-6", "cuda-cudart-dev-12-6", "cuda-nvml-dev-12-6"]);
    assert.deepEqual(record.cuda.licenses, cudaCopyrights, "every selected package's exact copyright bytes, owner and version are hash-bound");
    assert.equal(Object.hasOwn(record.cuda, "nvmlRuntime"), false, "the driver runtime is not copied into the input closure");
    assert.deepEqual(Object.keys(record.cuda.staticRuntime).sort(), ["cudart", "culibos"]);
    assert.equal(record.cuda.staticRuntime.cudart.path, path.join(state.cudaRoot, "lib64/libcudart_static.a"));
    assert.equal(record.cuda.staticRuntime.culibos.path, path.join(state.cudaRoot, "lib64/libculibos.a"));
    assert.deepEqual(record.cuda.fatbinary, {path: path.join(state.cudaRoot, "bin/fatbinary"), sha256: fileHash(path.join(state.cudaRoot, "bin/fatbinary"))});
    assert.ok(fs.statSync(record.cuda.fatbinary.path).mode & 0o111, "fatbinary is recorded as an executable tool input");
    assert.match(record.scope, /ptxas\/fatbinary inputs/);
    const missingFatbinary = {...record, cuda: Object.fromEntries(Object.entries(record.cuda).filter(([name]) => name !== "fatbinary"))};
    assert.throws(() => auditDpcppCudaInputs(state, missingFatbinary), /provenance is malformed or stale/);
    for (const obsolete of [missingFatbinary, {...record, scope: record.scope.replace(" and ptxas/fatbinary inputs", "")}]) {
      writeManifest(obsolete);
      assert.throws(() => auditDpcppCudaInputs(state, obsolete, fileHash(state.manifest)), /provenance is malformed or stale/,
        "a valid external pin must not admit an obsolete input schema");
    }
    writeManifest(record);
    put(state.cudaRoot, "bin/nvcc.profile", "TOP = /ambient/cuda\n");
    assert.throws(() => auditDpcppCudaInputs(state, record), /Selected DPC\+\+\/CUDA input changed/);
    put(state.cudaRoot, "bin/nvcc.profile", "TOP = $(_HERE_)/..\n");

    put(state.dpcppRoot, ".mom-toolchain-sha256", "0".repeat(64) + "\n");
    assert.throws(() => auditDpcppCudaInputs(state, record), /pinned DPC\+\+|Selected DPC\+\+\/CUDA/);
    put(state.dpcppRoot, ".mom-toolchain-sha256", dpcppPins.sdk + "\n");
    put(state.cudaRoot, "include/cuda.h", "changed header\n");
    assert.throws(() => auditDpcppCudaInputs(state, record), /headers are incomplete or changed/);
    put(state.cudaRoot, "include/cuda.h", "#define CUDA_VERSION 12060\n");
    assert.throws(() => auditDpcppCudaInputs(state, {...record, extra: true}), /malformed or stale/);
    put(state.cudaRoot, "lib64/libculibos.a", "changed archive\n");
    assert.throws(() => auditDpcppCudaInputs(state, record), /Selected DPC\+\+\/CUDA input changed/);
    put(state.cudaRoot, "lib64/libculibos.a", "!<arch>\n");
    put(state.cudaRoot, "bin/fatbinary", "changed fatbinary\n", 0o755);
    assert.throws(() => auditDpcppCudaInputs(state, record), /Selected DPC\+\+\/CUDA input changed: bin\/fatbinary/);
    put(state.cudaRoot, "bin/fatbinary", "fatbinary\n", 0o755);
    const originalCmakeEvidence = fs.readFileSync(state.cudaCmakeEvidence, "utf8");
    /** @param {(evidence:any)=>void} mutate */
    const rejectCmakeEvidence = mutate => {
      const evidence = JSON.parse(originalCmakeEvidence); mutate(evidence);
      const text = JSON.stringify(evidence, null, 2) + "\n";
      fs.writeFileSync(state.cudaCmakeEvidence, text);
      const sha256 = createHash("sha256").update(text).digest("hex"), altered = {...record, cuda: {...record.cuda,
        cmakeEvidence: {...record.cuda.cmakeEvidence, sha256}}};
      try {assert.throws(() => auditDpcppCudaInputs(state, altered), /successful same-root CMake toolkit smoke/);} finally {
        fs.writeFileSync(state.cudaCmakeEvidence, originalCmakeEvidence);
      }
    };
    rejectCmakeEvidence(evidence => {evidence.tool.path = "cmake";});
    rejectCmakeEvidence(evidence => {evidence.tool.sha256 = "not-a-sha256";});
    /** Re-pin each synthetic evidence mutation so these assertions exercise admission, not the manifest seal. */
    /** @param {(evidence:any)=>void} mutate @param {boolean} reject */
    const repinCmakeEvidence = (mutate, reject = true) => {
      const evidence = JSON.parse(originalCmakeEvidence); mutate(evidence);
      const text = JSON.stringify(evidence, null, 2) + "\n";
      fs.writeFileSync(state.cudaCmakeEvidence, text);
      const sha256 = createHash("sha256").update(text).digest("hex"), altered = {...record, cuda: {...record.cuda,
        cmakeEvidence: {...record.cuda.cmakeEvidence, sha256}}};
      writeManifest(altered); const pin = fileHash(state.manifest);
      try {
        if (reject) {
          assert.throws(() => auditDpcppCudaInputs(state, altered, pin), /successful same-root CMake toolkit smoke/);
        } else {assert.equal(auditDpcppCudaInputs(state, altered, pin), altered);}
      } finally {
        fs.writeFileSync(state.cudaCmakeEvidence, originalCmakeEvidence); writeManifest(record);
      }
      return evidence;
    };
    const producerBuild = path.join("/runner", path.basename(state.cudaCmakeBuild));
    const relocated = repinCmakeEvidence(evidence => {evidence.command[4] = producerBuild;}, false);
    assert.deepEqual(relocated.command, [cmake, "-S", state.cudaCmakeSource, "-B", producerBuild,
      "-DCUDAToolkit_ROOT=" + state.cudaRoot]);
    rejectCmakeEvidence(evidence => {evidence.command[4] = producerBuild;});
    for (const build of [path.join("/runner", path.basename(state.cudaCmakeBuild) + "-other"),
      path.basename(state.cudaCmakeBuild), "/runner/../" + path.basename(state.cudaCmakeBuild),
      path.join("/runner", "x".repeat(4100), path.basename(state.cudaCmakeBuild)), producerBuild + "\0suffix"]) {
      repinCmakeEvidence(evidence => {evidence.command[4] = build;});
    }
    repinCmakeEvidence(evidence => {evidence.command[2] = path.join(state.cudaCmakeSource, "other");});
    repinCmakeEvidence(evidence => {evidence.command[5] = "-DCUDAToolkit_ROOT=/other/cuda";});
    assert.throws(() => auditDpcppCudaInputs(state, {...record, cuda: {...record.cuda, root: "/tmp/stale-cuda"}}), /malformed or stale/);
    assert.throws(() => auditDpcppCudaInputs(state, {...record, cuda: {...record.cuda, cmakeEvidence: {...record.cuda.cmakeEvidence, path: "/tmp/stale-cmake.json"}}}), /malformed or stale/);

    assert.throws(() => auditDpcppCudaInputs(state, {...record, cuda: {...record.cuda, licenses: cudaCopyrights.slice(0, 1)}}), /copyrights are incomplete/);
    for (const row of cudaCopyrights) {
      const original = fs.readFileSync(row.path);
      fs.writeFileSync(row.path, "changed package terms\n");
      assert.throws(() => auditDpcppCudaInputs(state, record), /package copyright path, identity or body is missing or changed/);
      fs.writeFileSync(row.path, original);
      const altered = cudaCopyrights.map(item => item === row ? {...item, version: "12.6.0-1"} : item);
      assert.throws(() => auditDpcppCudaInputs(state, {...record, cuda: {...record.cuda, licenses: altered}}), /package copyright path, identity or body is missing or changed/);
    }

    /** Temporarily replace one required path with a symlink and require admission to reject it. */
    /** @param {string} source @param {string} backup @param {boolean} isDirectory */
    const assertSymlinkRejected = (source, backup, isDirectory) => {
      fs.renameSync(source, backup);
      try {
        fs.symlinkSync(backup, source, isDirectory ? "dir" : "file");
        assert.throws(() => auditDpcppCudaInputs(state, record), /symlink|unsafe|canonical|path/i);
      } finally {
        fs.unlinkSync(source); fs.renameSync(backup, source);
      }
    };
    assertSymlinkRejected(state.root, path.join(runnerTemp, "root-backup"), true);
    assertSymlinkRejected(path.join(state.cudaRoot, "include"), path.join(runnerTemp, "include-backup"), true);
    assertSymlinkRejected(path.join(state.cudaRoot, "bin/fatbinary"), path.join(runnerTemp, "fatbinary-backup"), false);
    assertSymlinkRejected(copyright, path.join(runnerTemp, "copyright-backup"), false);

    const wrongRoot = path.join(runnerTemp, "wrong"); fs.mkdirSync(wrongRoot); fs.mkdirSync(path.join(wrongRoot, "unexpected-source"));
    const wrongTar = path.join(runnerTemp, "wrong-source.tar"); packState(wrongTar, wrongRoot, ["unexpected-source"]);
    fs.mkdirSync(path.dirname(state.incomingSourceTar), {recursive: true}); fs.copyFileSync(wrongTar, state.incomingSourceTar);
    assert.throws(() => extractState(state.incomingSourceTar, runnerTemp, [state.linux.sourceName]), /unexpected state roots/);

    const copyRoot = path.join(runnerTemp, "cuda-source"), copyOut = path.join(runnerTemp, "cuda-out");
    put(copyRoot, "bin/nvcc", "nvcc\n", 0o755); put(copyRoot, "bin/ptxas", "ptxas\n", 0o755);
    put(copyRoot, "bin/fatbinary", "fatbinary payload\n", 0o755);
    put(copyRoot, "bin/nvcc.profile", "TOP = $(_HERE_)/..\n");
    put(copyRoot, "include/cuda.h", "cuda\n"); put(copyRoot, "nvvm/libdevice/libdevice.10.bc", "bc\n");
    put(copyRoot, "lib64/libcudart.so", "cudart\n");
    put(copyRoot, "lib64/libcudart_static.a", "!<arch>\n"); put(copyRoot, "lib64/libculibos.a", "!<arch>\n");
    const outside = put(runnerTemp, "outside-header.h", "outside\n"); fs.symlinkSync(outside, path.join(copyRoot, "include/escape.h"));
    assert.throws(() => copyCudaBuildInputs(copyRoot, copyOut), /symlink escapes/);

    const missingRoot = path.join(runnerTemp, "cuda-missing-fatbinary-source"), missingOut = path.join(runnerTemp, "cuda-missing-fatbinary-out");
    fs.cpSync(copyRoot, missingRoot, {recursive: true, verbatimSymlinks: true}); fs.unlinkSync(path.join(missingRoot, "bin/fatbinary"));
    assert.throws(() => copyCudaBuildInputs(missingRoot, missingOut), /** @param {NodeJS.ErrnoException} error */ error =>
      error.code === "ENOENT" && error.path?.endsWith(path.join("bin", "fatbinary")));

    const validRoot = path.join(runnerTemp, "cuda-valid-source"), validOut = path.join(runnerTemp, "cuda-valid-out");
    put(validRoot, "bin/nvcc", "nvcc\n", 0o755); put(validRoot, "bin/ptxas", "ptxas\n", 0o755);
    put(validRoot, "bin/fatbinary", "fatbinary payload\n", 0o755);
    put(validRoot, "bin/nvcc.profile", "TOP = $(_HERE_)/..\n");
    put(validRoot, "include/cuda.h", "cuda\n"); put(validRoot, "nvvm/libdevice/libdevice.10.bc", "bc\n");
    put(validRoot, "lib64/libcudart.so", "cudart\n"); put(validRoot, "lib64/libcudart_static.a", "!<arch>\n");
    put(validRoot, "lib64/libculibos.a", "!<arch>\n");
    copyCudaBuildInputs(validRoot, validOut);
    assert.equal(fs.readFileSync(path.join(validOut, "bin/nvcc.profile"), "utf8"), "TOP = $(_HERE_)/..\n");
    assert.equal(fs.readFileSync(path.join(validOut, "bin/fatbinary"), "utf8"), "fatbinary payload\n");
    assert.ok(fs.statSync(path.join(validOut, "bin/fatbinary")).mode & 0o111, "copy preserves the assembler executable bit");
    assert.equal(fs.readFileSync(path.join(validOut, "lib64/libcudart_static.a"), "utf8"), "!<arch>\n");
    assert.equal(fs.readFileSync(path.join(validOut, "lib64/libculibos.a"), "utf8"), "!<arch>\n");
  });

  test.describe("Pinned ROCm parser handoff", () => {
    const parserHelperPath = path.join(root, "scripts/native-ci-linux-sources.js");
    const parserProducerPath = path.join(root, "scripts/build-native-rocm-host-runtime.js");

    /** @param {import("node:test").TestContext} t */
    function fixture(t) {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-rocm-parser-"));
      t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
      const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(directory)},
        source = path.join("/tmp", "mom-native-rocm-prerequisites-" + suffix),
        state = {...rocmSourceStatePaths(paths, {MOM_NATIVE_ROCM_SOURCE_DIR: source}),
          pythonArchive: path.join(directory, "CppHeaderParser-2.7.4.tar.gz"),
          pythonSourceReceipt: path.join(directory, "receipt.json")},
        receipt = {version: 1, target: "rocm-parser-source", url: rocmParserSource.url,
          archive: {path: state.pythonArchive, bytes: rocmParserSource.bytes, sha256: rocmParserSource.sha256},
          helperSha256: fileHash(parserHelperPath), producerSha256: fileHash(parserProducerPath)};
      return {directory, state, receipt};
    }

    /** @param {{pythonSourceReceipt: string}} state @param {unknown} receipt */
    function writeReceipt(state, receipt) {
      fs.writeFileSync(state.pythonSourceReceipt, JSON.stringify(receipt) + "\n");
    }

    test("accepts a historical lower-case helper digest without mutating the receipt", t => {
      const {state, receipt} = fixture(t), current = receipt.helperSha256,
        historical = (current[0] === "0" ? "1" : "0") + current.slice(1), original = structuredClone(receipt);
      assert.match(historical, /^[a-f0-9]{64}$/);
      assert.notEqual(historical, current);
      receipt.helperSha256 = historical;
      assert.doesNotThrow(() => assertRocmParserReceipt(receipt, state.pythonArchive, receipt.producerSha256));
      assert.deepEqual(receipt, {...original, helperSha256: historical}, "receipt validation is read-only");
    });

    test("rejects missing or malformed acquisition-helper digests", t => {
      const {state, receipt} = fixture(t), malformed = [undefined, null, "", "not-a-hash", "a".repeat(63), "A".repeat(64), "a".repeat(65)];
      for (const helperSha256 of malformed) {
        const candidate = {...structuredClone(receipt), helperSha256};
        assert.throws(() => assertRocmParserReceipt(candidate, state.pythonArchive, receipt.producerSha256), /does not bind/);
      }
      const missing = /** @type {Partial<typeof receipt>} */ (structuredClone(receipt));
      delete missing.helperSha256;
      assert.throws(() => assertRocmParserReceipt(missing, state.pythonArchive, receipt.producerSha256), /does not bind/);
    });

    test("rejects metadata that no longer exactly binds the pinned parser archive", t => {
      const {state, receipt} = fixture(t), malformed = [
        {...structuredClone(receipt), extra: true},
        {...structuredClone(receipt), version: 2},
        {...structuredClone(receipt), target: "other-source"},
        {...structuredClone(receipt), url: "https://example.invalid/parser.tar.gz"},
        {...structuredClone(receipt), archive: {...receipt.archive, extra: true}},
        {...structuredClone(receipt), archive: {...receipt.archive, path: receipt.archive.path + ".other"}},
        {...structuredClone(receipt), archive: {...receipt.archive, bytes: receipt.archive.bytes + 1}},
        {...structuredClone(receipt), archive: {...receipt.archive, sha256: "0".repeat(64)}},
      ];
      for (const candidate of malformed) {
        assert.throws(() => assertRocmParserReceipt(candidate, state.pythonArchive, receipt.producerSha256), /does not bind/);
      }
    });

    test("rejects a receipt bound to a stale parser consumer", t => {
      const {state, receipt} = fixture(t), staleProducer = (receipt.producerSha256[0] === "0" ? "1" : "0") + receipt.producerSha256.slice(1);
      assert.throws(() => assertRocmParserReceipt(receipt, state.pythonArchive, staleProducer), /does not bind/);
    });

    test("rejects a parser archive with the wrong size", t => {
      const {state, receipt} = fixture(t);
      fs.writeFileSync(state.pythonArchive, Buffer.alloc(rocmParserSource.bytes - 1));
      writeReceipt(state, receipt);
      assert.throws(() => verifyRocmParserSource(state), /archive changed after acquisition/);
    });

    test("rejects same-size parser archive bytes that do not match the pinned digest", t => {
      const {state, receipt} = fixture(t);
      fs.writeFileSync(state.pythonArchive, Buffer.alloc(rocmParserSource.bytes, 0x5a));
      writeReceipt(state, receipt);
      assert.throws(() => verifyRocmParserSource(state), /archive changed after acquisition/);
    });

    test("rejects a parser archive supplied through a symlink", {skip: linuxOnlySkip}, t => {
      const {directory, state, receipt} = fixture(t), target = path.join(directory, "archive-target.tar.gz");
      fs.writeFileSync(target, Buffer.alloc(rocmParserSource.bytes));
      fs.symlinkSync(target, state.pythonArchive);
      writeReceipt(state, receipt);
      assert.throws(() => verifyRocmParserSource(state), /archive changed after acquisition/);
    });
  });

  test.describe("ROCm preinstalled host prerequisites", () => {
    test.describe("ROCm pkg-config host metadata", () => {
      test("selects the sysroot libelf/libnuma archives and the source-built libdrm SDK, never host /usr/lib", {skip: process.platform !== "linux"}, () => {
        /** @type {string[]} */ const admitted = [];
        const invoke = vm.runInNewContext("(" + rocmHostBuildConfig.toString() + ")", {path,
            requireLinuxSysroot: () => ({root: "/sysroot/root"}),
            nativeLibdrmInputs: (/** @type {string} */ root) => {
              admitted.push(root); return {root, drm: root + "/sdk/lib/libdrm.a",
                drmAmdgpu: root + "/sdk/lib/libdrm_amdgpu.a", pkgConfigDir: root + "/sdk/lib/pkgconfig"};
            }}),
          state = {source: "/tmp/mom-rocm-source", sdkRoot: "/tmp/mom-rocm-comgr", libdrmSdk: "/tmp/mom-rocm-host/libdrm"},
          tools = {ar: "/usr/bin/ar", cmake: "/usr/bin/cmake", dpkgQuery: "/usr/bin/dpkg-query", git: "/usr/bin/git",
            llvmObjcopy: "/tmp/mom-llvm/bin/llvm-objcopy", ninja: "/usr/bin/ninja", perl: "/usr/bin/perl",
            pkgConfig: "/usr/bin/pkg-config", python3: "/usr/bin/python3.12", ranlib: "/usr/bin/ranlib", tar: "/usr/bin/tar",
            xxd: "/usr/bin/xxd", cc: "/opt/intel/icx", cxx: "/opt/intel/icpx"};
        const config = invoke(state, tools, {});
        assert.deepEqual(admitted, [state.libdrmSdk]);
        assert.deepEqual({...config.dependencies}, {drm: "/tmp/mom-rocm-host/libdrm/sdk/lib/libdrm.a",
          drmAmdgpu: "/tmp/mom-rocm-host/libdrm/sdk/lib/libdrm_amdgpu.a", elf: "/sysroot/root/usr/lib/x86_64-linux-gnu/libelf.a",
          numa: "/sysroot/root/usr/lib/x86_64-linux-gnu/libnuma.a"});
        assert.equal(config.libdrm, state.libdrmSdk);
        assert.equal(config.pkgConfigDir, "/tmp/mom-rocm-host/libdrm/sdk/lib/pkgconfig");
        assert.doesNotMatch(JSON.stringify(config.dependencies), /"\/usr\/lib/);
      });
    });

    test("skips APT only for all-installed packages and exact PLY/xxd/OpenGL development pins", () => {
      const expected = {"libdrm-dev": "2.4.120-1", meson: "1.3.2-1ubuntu1",
        "python3-ply": "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20", "libegl-dev": "1.7.0-1build1",
        "libopengl-dev": "1.7.0-1build1"};
      const pinned = {"python3-ply": "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20",
        "libegl-dev": "1.7.0-1build1", "libopengl-dev": "1.7.0-1build1"};
      /** @type {Array<{command:string,args:string[]}>} */ const calls = [];
      /** @type {Record<string,string>} */ let selected = expected;
      /** @type {string|undefined} */ let dpkgOutput;
      const rows = () => Object.entries(selected).map(([name, version]) => `${name}\t${version}\tii `).join("\n") + "\n";
      /** @param {string} stdout @param {number} [status] @returns {ReturnType<typeof spawnSync>} */
      const result = (stdout, status = 0) => /** @type {ReturnType<typeof spawnSync>} */ ({
        pid: 1, output: [null, stdout, ""], stdout, stderr: "", status, signal: null});
      const runner = /** @type {typeof spawnSync} */ (/** @type {unknown} */ (
        /** @param {string} command @param {string[]} args */
        (command, args) => {
          calls.push({command, args});
          if (command === "/usr/bin/dpkg-query") {return result(dpkgOutput ?? rows());}
          if (command === "/usr/bin/apt-cache") {
            return result(Object.entries(pinned).map(([name, version]) => name + " | " + version + "\n").join(""));
          }
          return result("");
        }));
      const admitted = provisionRocmHostPackages({env: {}}, runner);
      assert.ok(admitted);
      assert.equal(admitted.mode, "already-installed");
      assert.deepEqual(calls.map(row => row.command), ["/usr/bin/dpkg-query"], "a valid package inventory never invokes sudo/APT");
      for (const [name, version] of Object.entries({"python3-ply": "3.11-5", xxd: "2:9.1.0016-1ubuntu6",
        "libegl-dev": "1.7.0-1build0", "libopengl-dev": "1.7.0-1build0"})) {
        selected = {...expected, [name]: version}; dpkgOutput = undefined; calls.length = 0;
        assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
        assert.deepEqual(calls.map(row => row.command), ["/usr/bin/dpkg-query", "sudo", "/usr/bin/apt-cache", "sudo"],
          `${name} version mismatch keeps the existing pinned APT fallback`);
      }
      selected = {...expected}; dpkgOutput = rows().replace("libdrm-dev\t2.4.120-1\tii ", "libdrm-dev\t2.4.120-1\tdeinstall ");
      calls.length = 0;
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
      assert.ok(calls.some(row => row.command === "sudo"), "non-installed package status falls back to APT");
      dpkgOutput = rows() + "libdrm-dev\t2.4.120-1\tii \n";
      calls.length = 0;
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
      assert.ok(calls.some(row => row.command === "sudo"), "duplicate package rows fall back to APT");
      dpkgOutput = rows() + "rogue-package\t1.0\tii \n";
      calls.length = 0;
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
      assert.ok(calls.some(row => row.command === "sudo"), "extra package rows fall back to APT");
      dpkgOutput = rows().replace("libdrm-dev\t2.4.120-1\tii ", "libdrm-dev\t\tii ");
      calls.length = 0;
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
      assert.ok(calls.some(row => row.command === "sudo"), "malformed package rows fall back to APT");
      selected = {...expected}; delete selected["meson"]; dpkgOutput = undefined; calls.length = 0;
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined);
      assert.ok(calls.some(row => row.command === "sudo"), "an incomplete package inventory falls back to APT");
    });

    test("an older package inventory cannot skip required EGL/OpenGL development packages", () => {
      const oldPackages = {"libdrm-dev": "2.4.120-1", meson: "1.3.2-1ubuntu1",
        "python3-ply": "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20"};
      const pinned = {"python3-ply": "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20",
        "libegl-dev": "1.7.0-1build1", "libopengl-dev": "1.7.0-1build1"};
      /** @type {Array<{command:string,args:string[]}>} */ const calls = [];
      /** @param {string} stdout @returns {ReturnType<typeof spawnSync>} */
      const result = stdout => /** @type {ReturnType<typeof spawnSync>} */ ({
        pid: 1, output: [null, stdout, ""], stdout, stderr: "", status: 0, signal: null});
      const runner = /** @type {typeof spawnSync} */ (/** @type {unknown} */ (
        /** @param {string} command @param {string[]} args */
        (command, args) => {
          calls.push({command, args});
          if (command === "/usr/bin/dpkg-query") {
            return result(Object.entries(oldPackages).map(([name, version]) => name + "\t" + version + "\tii ").join("\n") + "\n");
          }
          if (command === "/usr/bin/apt-cache") {
            return result(Object.entries(pinned).map(([name, version]) => name + " | " + version + "\n").join(""));
          }
          return result("");
        }));
      assert.equal(provisionRocmHostPackages({env: {}}, runner), undefined,
        "an old image missing EGL/OpenGL headers must take the existing pinned APT path");
      assert.deepEqual(calls.map(row => row.command), ["/usr/bin/dpkg-query", "sudo", "/usr/bin/apt-cache", "sudo"]);
      assert.deepEqual(calls[2]?.args, ["madison", ...Object.keys(pinned)]);
      const install = calls[3];
      assert.ok(install);
      for (const [name, version] of Object.entries(pinned)) {
        assert.ok(install.args.includes(name + "=" + version), "APT install must pin " + name);
      }
    });

    test("reuses only a canonical exact CppHeaderParser file tree", {skip: linuxOnlySkip}, t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-parser-preinstalled-")),
        expected = /** @type {Record<string, string>} */ ({}), source = path.join(root, "source"),
        destination = path.join(root, "install", "CppHeaderParser");
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      fs.mkdirSync(source, {recursive: true});
      for (const [name, contents] of Object.entries({"__init__.py": "__version__ = '2.7.4'\n", "CppHeaderParser.py": "PINNED\n"})) {
        fs.writeFileSync(path.join(source, name), contents);
        expected[name] = fileHash(path.join(source, name));
      }
      assert.equal(admitRocmParserInstallation(destination, expected), undefined, "absent target follows unchanged installer path");
      fs.mkdirSync(destination, {recursive: true});
      for (const name of Object.keys(expected)) {fs.copyFileSync(path.join(source, name), path.join(destination, name));}
      assert.deepEqual(admitRocmParserInstallation(destination, expected), expected);
      fs.writeFileSync(path.join(destination, "extra.py"), "extra\n");
      assert.throws(() => admitRocmParserInstallation(destination, expected), /differ from the pinned source distribution/);
      fs.unlinkSync(path.join(destination, "extra.py"));
      fs.writeFileSync(path.join(destination, "CppHeaderParser.py"), "changed\n");
      assert.throws(() => admitRocmParserInstallation(destination, expected), /differ from the pinned source distribution/);
      fs.copyFileSync(path.join(source, "CppHeaderParser.py"), path.join(destination, "CppHeaderParser.py"));
      fs.unlinkSync(path.join(destination, "__init__.py"));
      assert.throws(() => admitRocmParserInstallation(destination, expected), /differ from the pinned source distribution/);
      fs.copyFileSync(path.join(source, "__init__.py"), path.join(destination, "__init__.py"));
      fs.symlinkSync(path.join(source, "CppHeaderParser.py"), path.join(destination, "linked.py"));
      assert.throws(() => admitRocmParserInstallation(destination, expected), /regular files/);
      fs.unlinkSync(path.join(destination, "linked.py"));
      const alias = path.join(root, "alias"); fs.symlinkSync(destination, alias, "dir");
      assert.throws(() => admitRocmParserInstallation(alias, expected), /noncanonical or redirected/);
    });
  });

  test("ROCm source handoff binds pinned roots and parser sdist without rewriting the Git receipt", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-rocm-source-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const source = "/tmp/mom-native-rocm-prerequisites-123456-2", git = "/usr/bin/git",
      gitSha256 = "a".repeat(64), fetcherSha256 = "b".repeat(64), selected = gitSourceComponents("rocm"), recipe = sourceRecipe("rocm"),
      names = /** @type {(keyof typeof origins)[]} */ (Object.keys(selected).sort()), roots = Object.fromEntries(names.map(name => [name, path.join(source, "roots", name)])),
      components = Object.fromEntries(names.map(name => {
        const component = selected[name], origin = origins[name]; assert.ok(component); assert.ok(origin);
        return [name, {origin, revision: component.revision, root: roots[name], gitDirectory: "bare", shallow: true, partialClone: false}];
      })),
      receipt = {version: 1, target: "rocm", status: "PASS", networked: true, buildOnly: true, compiled: false,
        publicationQualified: false, git: {path: git, sha256: gitSha256}, fetcherSha256,
        recipe: {sha256: createHash("sha256").update(JSON.stringify(recipe)).digest("hex"), files: recipe.graph}, components},
      original = structuredClone(receipt), paths = {suffix: "123456-2", runnerTemp: fs.realpathSync(runnerTemp)},
      state = rocmSourceStatePaths(paths, {MOM_NATIVE_ROCM_SOURCE_DIR: source});
    assert.deepEqual(names, ["clr", "hip", "rocmLlvm", "rocr"]);
    assert.equal(state.sourceName, "mom-native-rocm-prerequisites-123456-2");
    assert.equal(state.source, source);
    assert.equal(state.rootsFile, path.join(source, "roots.json"));
    assert.equal(state.fetchReceipt, path.join(source, "fetch.json"));
    assert.equal(state.pythonArchive, path.join(source, "python-source/CppHeaderParser-2.7.4.tar.gz"));
    assert.equal(state.pythonSourceReceipt, path.join(source, "python-source/receipt.json"));
    assert.equal(path.basename(state.sourceTar), "native-rocm-state-123456-2-sources.tar");
    assert.equal(state.sourceTar, path.join(runnerTemp, "native-rocm-state-123456-2-sources.tar"));
    assert.equal(state.incomingTar, path.join(runnerTemp, "incoming/native-rocm-state-123456-2-sources.tar"));
    assert.deepEqual(rocmSourceFetchArgs(state, git), ["rocm", git, source, "--allow-network"]);
    assert.ok(Object.hasOwn(recipe.graph, "scripts/build-native-comgr-sdk.js"));
    assert.ok(Object.hasOwn(recipe.graph, "scripts/build-native-rocm-host-runtime.js"));
    assert.ok(Object.hasOwn(recipe.graph, "scripts/build-native-rocm-resources.js"));
    assert.doesNotThrow(() => assertSdkFetchProvenance(receipt, roots, "rocm", source, fetcherSha256, {path: git, sha256: gitSha256}));
    assert.deepEqual(receipt, original, "source admission does not mutate the immutable fetch receipt");
    for (const altered of [
      {...structuredClone(receipt), target: "compiler"},
      {...structuredClone(receipt), publicationQualified: true},
      {...structuredClone(receipt), git: {...receipt.git, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), recipe: {...receipt.recipe, sha256: "0".repeat(64)}},
      {...structuredClone(receipt), components: {...receipt.components, rocmLlvm: {...receipt.components["rocmLlvm"], revision: "0".repeat(40)}}},
      {...structuredClone(receipt), components: {...receipt.components, clr: {...receipt.components["clr"], origin: "https://example.invalid/clr.git"}}},
      {...structuredClone(receipt), components: {...receipt.components, hip: {...receipt.components["hip"], root: "/tmp/ambient/hip"}}},
      {...structuredClone(receipt), components: Object.fromEntries(Object.entries(receipt.components).filter(([name]) => name !== "rocr"))},
    ]) {assert.throws(() => assertSdkFetchProvenance(altered, roots, "rocm", source, fetcherSha256, {path: git, sha256: gitSha256}), /does not bind|changed its pinned/);}
    assert.throws(() => assertSdkFetchProvenance(receipt, {...roots, extra: source}, "rocm", source, fetcherSha256), /does not bind/);
    assert.throws(() => assertSdkFetchProvenance(receipt, {...roots, rocmLlvm: "/tmp/ambient/rocm"}, "rocm", source, fetcherSha256), /changed its pinned/);
    assert.throws(() => rocmSourceStatePaths(paths, {MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/ambient-rocm"}), /exactly match/);
  });

  test("ROCm/COMGR build handoff consumes the exact same-run source and LLVM SDK roots", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-rocm-comgr-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp),
        source: "/tmp/mom-native-prerequisites-" + suffix, sourceName: "mom-native-prerequisites-" + suffix,
        ioSourceName: "mom-native-io-prerequisites-" + suffix,
        sdkRoot: "/tmp/mom-native-llvm21-sdk-" + suffix, sdkName: "mom-native-llvm21-sdk-" + suffix,
        output: "/tmp/mom-native-llvm21-sdk-" + suffix + "/linux-llvm21-sdk"},
      environment = {MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
        MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix},
      state = rocmComgrStatePaths(paths, environment), tools = {cc: "/tools/icx", cxx: "/tools/icpx", cmake: "/tools/cmake"};
    assert.equal(state.incomingSourceTar, path.join(runnerTemp, "incoming/native-rocm-state-123456-2-sources.tar"));
    assert.deepEqual(state.llvmSourceArchiveRoots, [paths.sourceName, paths.ioSourceName]);
    assert.equal(state.incomingLlvmSdkTar, path.join(runnerTemp, "incoming/native-linux-llvm21-installed-sdk-123456-2.tar"));
    assert.equal(state.incomingLlvmSourceTar, path.join(runnerTemp, "incoming/native-llvm21-state-123456-2-sources.tar"));
    assert.equal(state.llvmSdk, "/tmp/mom-native-llvm21-sdk-123456-2/linux-llvm21-sdk/sdk");
    assert.equal(state.sdkTar, path.join(runnerTemp, "native-linux-rocm-comgr-sdk-123456-2.tar"));
    assert.equal(rocmClrHipamdTreeish("5b591b2ff64aac4c1c08be879b0d5f89fd713d50"), "5b591b2ff64aac4c1c08be879b0d5f89fd713d50:hipamd");
    assert.throws(() => rocmClrHipamdTreeish("clr-hipamd"), /Invalid pinned CLR revision/);
    const headerPatch = fs.readFileSync(path.join(root, "scripts/patches/rocm-hip-headers-only.patch"), "utf8");
    assert.equal(createHash("sha256").update(headerPatch).digest("hex"), "caed551ccaa0d8c3469676d4e3a74c6cd353ed75425297398119d36eca3854e0");
    assert.match(headerPatch, /^\+if\(NOT WIN32 AND NOT HIP_RUNTIME STREQUAL "none"\)$/m);
    assert.doesNotMatch(headerPatch, /^\+if\(NOT WIN32\)$/m);
    const config = rocmResourcesConfig(state, tools);
    assert.deepEqual(config, {version: 1, target: "linux", source: environment.MOM_NATIVE_ROCM_SOURCE_DIR + "/roots/rocmLlvm",
      clr: environment.MOM_NATIVE_ROCM_SOURCE_DIR + "/roots/clr", llvm: state.llvmSdk, cmake: tools.cmake,
      hipHeaders: state.hipIncludeRoot, hostCompilers: {cc: tools.cc, cxx: tools.cxx}});
    assert.deepEqual(rocmResourcesConfig(state, tools, "win32"), {...config, target: "win32"});
    assert.throws(() => rocmResourcesConfig(state, tools, /** @type {any} */ ("windows")), /Unsupported ROCm resources target/);
    assert.deepEqual(rocmComgrBuildCommands(state, tools), [
      {script: "scripts/build-native-rocm-resources.js", args: [state.resourcesConfig, state.resourcesRoot], config},
      {script: "scripts/build-native-comgr-sdk.js", args: [environment.MOM_NATIVE_ROCM_SOURCE_DIR + "/roots/rocmLlvm",
        state.llvmSdk, state.llvmSdk, state.resourcesRoot + "/install", state.sdkRoot, tools.cc, tools.cxx]},
    ]);
    assert.throws(() => rocmComgrStatePaths(paths, {...environment, MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/ambient-rocm-sdk"}), /exactly match/);
  });

  test("ROCm host SDK output and COMGR input paths are exact per run attempt", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-rocm-host-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp),
        source: "/tmp/mom-native-prerequisites-" + suffix, sourceName: "mom-native-prerequisites-" + suffix,
        ioSourceName: "mom-native-io-prerequisites-" + suffix,
        sdkRoot: "/tmp/mom-native-llvm21-sdk-" + suffix, sdkName: "mom-native-llvm21-sdk-" + suffix,
        output: "/tmp/mom-native-llvm21-sdk-" + suffix + "/linux-llvm21-sdk"},
      environment = {MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
        MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix,
        MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/mom-native-rocm-host-sdk-" + suffix}, state = rocmHostStatePaths(paths, environment);
    assert.equal(state.hostSdkName, "mom-native-rocm-host-sdk-123456-2");
    assert.equal(state.hostSdkRoot, environment.MOM_NATIVE_ROCM_HOST_SDK_DIR);
    assert.equal(state.runtimeOutput, path.join(state.hostSdkRoot, "runtime"));
    assert.equal(state.configFile, path.join(state.hostSdkRoot, "rocm-host.json"));
    assert.equal(state.incomingComgrTar, path.join(runnerTemp, "incoming/native-linux-rocm-comgr-sdk-123456-2.tar"));
    assert.equal(state.hostSdkTar, path.join(runnerTemp, "native-linux-rocm-host-sdk-123456-2.tar"));
    assert.equal(state.llvmSdk, path.join(paths.output, "sdk"));
    assert.throws(() => rocmHostStatePaths(paths, {...environment, MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/ambient"}), /exactly match/);
  });

  test("ROCm HIP headers include pinned common, CLR AMD, and generated version files", {skip: linuxOnlySkip}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-hip-headers-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const common = path.join(root, "hip-common/include/hip"), clr = path.join(root, "clr-hipamd/include/hip/amd_detail"),
      buildInclude = path.join(root, "build/include/hip"), output = path.join(root, "package");
    /** @param {string} file @param {string} [value] */
    const write = (file, value = "header\n") => {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value);};
    /** @type {string[]} */ const rows = [];
    for (let index = 0; index < 12; index++) {
      const name = `common_${index}.h`;
      write(path.join(common, name)); rows.push(`\${HIP_COMMON_INCLUDE_DIR}/hip/${name}`);
    }
    for (let index = 0; index < 4; index++) {
      const name = `clr_${index}.h`;
      write(path.join(clr, name)); rows.push(`\${PROJECT_SOURCE_DIR}/include/hip/amd_detail/${name}`);
    }
    const cmake = `set(HIPRTC_HEADERS\n${rows.join("\n")}\n)\n`, generated = path.join(buildInclude, "hip_version.h");
    write(generated, "#define HIP_VERSION_MAJOR 7\n");
    const result = assembleRocmHipHeaders(common.slice(0, -"/include/hip".length),
      clr.slice(0, -"/include/hip/amd_detail".length), generated, output, cmake);
    assert.deepEqual(result.required, hiprtcHeaders(cmake, output));
    assert.equal(Object.keys(result.files).length, 17);
    assert.equal(fs.readFileSync(path.join(output, "hip/amd_detail/clr_3.h"), "utf8"), "header\n");
    assert.equal(fs.readFileSync(path.join(output, "hip/hip_version.h"), "utf8"), "#define HIP_VERSION_MAJOR 7\n");

    const partialClr = path.join(root, "partial-clr/include/hip/amd_detail");
    write(path.join(partialClr, "clr_0.h"));
    assert.throws(() => assembleRocmHipHeaders(common.slice(0, -"/include/hip".length),
      partialClr.slice(0, -"/include/hip/amd_detail".length), generated, path.join(root, "missing-clr-package"), cmake),
    /Required HIPRTC header is missing/);
    assert.throws(() => assembleRocmHipHeaders(common.slice(0, -"/include/hip".length),
      clr.slice(0, -"/include/hip/amd_detail".length), path.join(root, "missing/hip_version.h"),
      path.join(root, "missing-version-package"), cmake), /lacks common, CLR or generated-version inputs/);

    const collisionCommon = path.join(root, "collision/include/hip");
    write(path.join(collisionCommon, "normal.h")); write(path.join(collisionCommon, "amd_detail/unexpected.h"));
    assert.throws(() => assembleRocmHipHeaders(root + "/collision", clr.slice(0, -"/include/hip/amd_detail".length),
      generated, path.join(root, "collision-package"), cmake), /lacks common, CLR or generated-version inputs/);
    const suffix = "123456-2", runnerTemp = path.join(root, "runner"); fs.mkdirSync(runnerTemp);
    const paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp), source: "/tmp/mom-native-rocm-source-" + suffix,
        sourceName: "mom-native-rocm-source-" + suffix, ioSourceName: "mom-native-io-prerequisites-" + suffix,
        sdkRoot: "/tmp/mom-native-llvm21-sdk-" + suffix,
        sdkName: "mom-native-llvm21-sdk-" + suffix, output: "/tmp/mom-native-llvm21-sdk-" + suffix + "/linux-llvm21-sdk"},
      environment = {MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
        MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix},
      state = rocmComgrStatePaths(paths, environment);
    assert.deepEqual(hipHeaderConfigureCommand(state, {cc: "/tools/icx", cxx: "/tools/icpx", cmake: "/tools/cmake"}), [
      "/tools/cmake", "-S", state.clrHipamdRoot, "-B", state.hipBuildRoot, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_INSTALL_PREFIX=" + state.hipInstallPrefix, "-DCMAKE_C_COMPILER=/tools/icx", "-DCMAKE_CXX_COMPILER=/tools/icpx",
      "-DHIP_COMMON_DIR=" + state.hipCommonRoot, "-DHIP_PLATFORM:STRING=amd", "-DHIP_RUNTIME:STRING=none",
      "-DHIPCC_BIN_DIR:STRING=", "-DHIP_OFFICIAL_BUILD:BOOL=ON", "-DROCM_PATH=" + state.hipInstallPrefix,
    ]);
  });

  test("Hermes run roots, input/output tar paths, and stage budgets are exact and bounded", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-hermes-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123456-2", runnerTemp: fs.realpathSync(runnerTemp)},
      environment = {MOM_NATIVE_HERMES_SOURCE_DIR: "/tmp/mom-native-hermes-prerequisites-123456-2",
        MOM_NATIVE_HERMES_SDK_DIR: "/tmp/mom-native-hermes-sdk-123456-2"}, state = hermesStatePaths(paths, environment);
    assert.equal(state.sourceName, "mom-native-hermes-prerequisites-123456-2");
    assert.equal(state.sdkName, "mom-native-hermes-sdk-123456-2");
    assert.equal(path.dirname(state.source), "/tmp");
    assert.equal(path.dirname(state.sdkRoot), "/tmp");
    assert.equal(path.basename(state.sourceTar), "native-hermes-state-123456-2-sources.tar");
    assert.equal(path.basename(state.incomingTar), "native-hermes-state-123456-2-sources.tar");
    assert.equal(path.basename(state.sdkTar), "native-linux-hermes-sdk-123456-2.tar");
    const tools = {git: "/tools/git", cmake: "/tools/cmake", cc: "/tools/icx", cxx: "/tools/icpx", ninja: "/tools/ninja"},
      command = hermesBuildCommand(state, tools, "4");
    assert.equal(command.script, "scripts/build-native-hermes.js");
    assert.equal(command.args[0], path.join(state.source, "roots/hermes"), "the producer receives the fetched bare Git root");
    assert.notEqual(command.args[0], state.source, "the parent fetch cohort is not a Git repository");
    assert.deepEqual(command.args.slice(1), [state.sdkRoot, tools.git, tools.cmake, tools.cc, tools.cxx, tools.ninja, "4"]);
    assert.throws(() => hermesStatePaths(paths, {...environment, MOM_NATIVE_HERMES_SDK_DIR: "/tmp/ambient-hermes-sdk"}), /exactly match/);
  });

  test("Hermes SDK fingerprints accept only one canonical producer patch prefix", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-hermes-fingerprints-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const tools = /** @type {ReturnType<typeof import("../scripts/native-ci-linux-support").hermesBuildTools>} */ ({
        git: path.join(root, "git"), cmake: path.join(root, "cmake"), cc: path.join(root, "icx"),
        cxx: path.join(root, "icpx"), ninja: path.join(root, "ninja")}), relative = "runtime/hermes-host-rejections.patch",
      patch = path.join(root, relative), producerPatch = "/original/checkout/" + relative;
    for (const file of Object.values(tools)) {
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, path.basename(file) + "\n");
    }
    fs.mkdirSync(path.dirname(patch), {recursive: true});
    fs.copyFileSync(path.join(__dirname, "../runtime/hermes-host-rejections.patch"), patch);
    const fingerprints = Object.fromEntries([...Object.values(tools), patch].map(file => [file, fileHash(file)]));
    const originalPathFingerprints = {...fingerprints}; delete originalPathFingerprints[patch];
    originalPathFingerprints[producerPatch] = fileHash(patch);
    assert.doesNotThrow(() => assertHermesSdkFingerprints(originalPathFingerprints, tools, [relative]));

    /** @param {(changed: Record<string, string>) => void} edit @param {ReturnType<typeof import("../scripts/native-ci-linux-support").hermesBuildTools>} [editTools]
     * @param {string[]} [editPatches] */
    const reject = (edit, editTools = tools, editPatches = [relative]) => {
      const changed = {...originalPathFingerprints}; edit(changed);
      assert.throws(() => assertHermesSdkFingerprints(changed, editTools, editPatches), /Hermes SDK/);
    };
    reject(changed => {delete changed[producerPatch];});
    reject(changed => {changed["/second/checkout/" + relative] = fileHash(patch);});
    reject(changed => {delete changed[producerPatch]; changed["/original/checkout/../checkout/" + relative] = fileHash(patch);});
    reject(changed => {delete changed[producerPatch]; changed["/original/checkout/\u0001/" + relative] = fileHash(patch);});
    reject(changed => {changed[producerPatch] = "0".repeat(64);});
    reject(changed => {changed["/unexpected/tool"] = "1".repeat(64);});
    reject(changed => {changed[tools.cmake] = "2".repeat(64);});
    const extraTools = {...tools, ar: path.join(root, "ar")};
    assert.throws(() => assertHermesSdkFingerprints(originalPathFingerprints, extraTools, [relative]), /Hermes SDK/);
    assert.throws(() => assertHermesSdkFingerprints(originalPathFingerprints, tools, [relative, "runtime/another.patch"]), /Hermes SDK/);
  });

  test("run-scoped I/O roots are canonical and remain separate from LLVM artifacts", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const names = ["MOM_NATIVE_CI_SUFFIX", "RUNNER_TEMP", "MOM_AOT_BUILD_ONLY", "MOM_BUILD_JOBS",
      "MOM_NATIVE_SOURCE_DIR", "MOM_NATIVE_LINUX_SDK_DIR", "MOM_NATIVE_IO_SOURCE_DIR", "MOM_NATIVE_IO_SDK_DIR",
      "MOM_NATIVE_LINUX_SOURCES_DIR", "MOM_NATIVE_LINUX_JIT_RESOURCES_DIR", "MOM_NATIVE_ZSTD_SOURCE_DIR", "MOM_NATIVE_ZSTD_SDK_DIR", "MOM_NATIVE_ROCM_COMGR_SDK_DIR",
      "MOM_NATIVE_ROCM_RESOURCES_DIR", "MOM_NATIVE_ROCM_HIP_HEADERS_DIR", "MOM_NATIVE_ROCM_HOST_SDK_DIR", "MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR"];
    const before = Object.fromEntries(names.map(name => [name, process.env[name]]));
    try {
      Object.assign(process.env, {MOM_NATIVE_CI_SUFFIX: "123456-2", RUNNER_TEMP: runnerTemp, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "4",
        MOM_NATIVE_SOURCE_DIR: "/tmp/mom-native-prerequisites-123456-2", MOM_NATIVE_LINUX_SDK_DIR: "/tmp/mom-native-llvm21-sdk-123456-2",
        MOM_NATIVE_IO_SOURCE_DIR: "/tmp/mom-native-io-prerequisites-123456-2", MOM_NATIVE_IO_SDK_DIR: "/tmp/mom-native-io-openssl-sdk-123456-2",
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-123456-2",
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-123456-2",
        MOM_NATIVE_ZSTD_SOURCE_DIR: "/tmp/mom-native-zstd-prerequisites-123456-2", MOM_NATIVE_ZSTD_SDK_DIR: "/tmp/mom-native-zstd-sdk-123456-2",
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-123456-2", MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-123456-2",
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-123456-2",
        MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/mom-native-rocm-host-sdk-123456-2",
        MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: "/tmp/mom-native-linux-sdk-descriptor-123456-2"});
      const paths = ciPaths();
      assert.equal(paths.linuxSource, "/tmp/mom-native-linux-source-prerequisites-123456-2");
      assert.equal(paths.ioSource, "/tmp/mom-native-io-prerequisites-123456-2");
      assert.equal(paths.ioSdkRoot, "/tmp/mom-native-io-openssl-sdk-123456-2");
      assert.equal(paths.linuxSdkDescriptorRoot, "/tmp/mom-native-linux-sdk-descriptor-123456-2");
      assert.equal(path.basename(ioFinalTar(paths)), "native-linux-io-openssl-sdk-123456-2.tar");
      assert.notEqual(paths.ioSdkRoot, paths.sdkRoot);
      assert.deepEqual(ioOpenSslBuildCommands(paths, "/tool/cmake", "/tool/ninja", {
        ar: "/tool/ar", cc: "/tool/icx", make: "/tool/make", perl: "/tool/perl", ranlib: "/tool/ranlib", tar: "/tool/tar",
      }, "4"), [
        {script: "scripts/build-native-io.js", args: ["/tmp/mom-native-io-prerequisites-123456-2/libuv-v1.51.0.tar.gz",
          "/tmp/mom-native-io-prerequisites-123456-2/zlib-1.3.1.tar.gz", "/tmp/mom-native-io-openssl-sdk-123456-2/io-sdk",
          "/tool/cmake", "/tool/icx", "/tool/ninja", "4"]},
        {script: "scripts/build-native-openssl.js", args: ["/tmp/mom-native-prerequisites-123456-2/openssl-3.5.9.tar.gz",
          "/tmp/mom-native-io-openssl-sdk-123456-2/openssl-sdk", "/tmp/mom-native-io-openssl-sdk-123456-2/openssl-tools.json"]},
      ]);
    } finally {
      for (const name of names) {if (before[name] === undefined) {delete process.env[name];} else {process.env[name] = before[name];}}
    }
  });

  test("zstd source and SDK handoffs preserve the producer root and pinned original path", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-zstd-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const names = ["MOM_NATIVE_CI_SUFFIX", "RUNNER_TEMP", "MOM_AOT_BUILD_ONLY", "MOM_BUILD_JOBS",
      "MOM_NATIVE_SOURCE_DIR", "MOM_NATIVE_LINUX_SDK_DIR", "MOM_NATIVE_IO_SOURCE_DIR", "MOM_NATIVE_IO_SDK_DIR",
      "MOM_NATIVE_LINUX_SOURCES_DIR", "MOM_NATIVE_LINUX_JIT_RESOURCES_DIR", "MOM_NATIVE_ZSTD_SOURCE_DIR", "MOM_NATIVE_ZSTD_SDK_DIR", "MOM_NATIVE_ROCM_COMGR_SDK_DIR",
      "MOM_NATIVE_ROCM_RESOURCES_DIR", "MOM_NATIVE_ROCM_HIP_HEADERS_DIR", "MOM_NATIVE_ROCM_HOST_SDK_DIR", "MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR"];
    const before = Object.fromEntries(names.map(name => [name, process.env[name]]));
    try {
      Object.assign(process.env, {MOM_NATIVE_CI_SUFFIX: "123456-2", RUNNER_TEMP: runnerTemp, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "4",
        MOM_NATIVE_SOURCE_DIR: "/tmp/mom-native-prerequisites-123456-2", MOM_NATIVE_LINUX_SDK_DIR: "/tmp/mom-native-llvm21-sdk-123456-2",
        MOM_NATIVE_IO_SOURCE_DIR: "/tmp/mom-native-io-prerequisites-123456-2", MOM_NATIVE_IO_SDK_DIR: "/tmp/mom-native-io-openssl-sdk-123456-2",
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-123456-2",
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-123456-2",
        MOM_NATIVE_ZSTD_SOURCE_DIR: "/tmp/mom-native-zstd-prerequisites-123456-2", MOM_NATIVE_ZSTD_SDK_DIR: "/tmp/mom-native-zstd-sdk-123456-2",
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-123456-2", MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-123456-2",
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-123456-2",
        MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/mom-native-rocm-host-sdk-123456-2",
        MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: "/tmp/mom-native-linux-sdk-descriptor-123456-2"});
      const paths = ciPaths(), state = zstdStatePaths(paths), catalog = zstdArchiveCatalog();
      assert.equal(state.source, "/tmp/mom-native-zstd-prerequisites-123456-2");
      assert.equal(state.sdkRoot, "/tmp/mom-native-zstd-sdk-123456-2");
      assert.equal(state.archive, path.join(state.source, "zstd-" + zstdPin.version + ".tar.gz"));
      assert.equal(state.sourceTar, path.join(runnerTemp, "native-zstd-state-123456-2-sources.tar"));
      assert.equal(state.incomingTar, path.join(runnerTemp, "incoming/native-zstd-state-123456-2-sources.tar"));
      assert.equal(state.sdkTar, path.join(runnerTemp, "native-linux-zstd-sdk-123456-2.tar"));
      assert.equal(catalog["zstd"]?.sha256, zstdPin.sha256);
      assert.equal(catalog["zstd"]?.fileName, "zstd-" + zstdPin.version + ".tar.gz");
      assert.deepEqual(Object.keys(zstdSourceRecipe().graph), ["scripts/build-native-zstd.js", "scripts/build-source-inputs.js",
        "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-linux-sysroot.js"]);
      const command = zstdBuildCommand(state, state.toolsFile);
      assert.deepEqual(command, {script: "scripts/build-native-zstd.js", args: [state.archive, state.sdkRoot, state.toolsFile]});
      assert.notEqual(command.args[1], path.join(state.sdkRoot, "lib/libzstd.a"), "the builder receives SDKROOT, not a raw archive path");
    } finally {
      for (const name of names) {if (before[name] === undefined) {delete process.env[name];} else {process.env[name] = before[name];}}
    }
  });

  test("zstd audit joins the verified original archive hash to nativeZstdInputs(SDKROOT)", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-zstd-audit-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const source = path.join(root, "mom-native-zstd-source"), sdkRoot = path.join(root, "mom-native-zstd-sdk"),
      archive = path.join(source, "zstd-" + zstdPin.version + ".tar.gz"), receipt = path.join(sdkRoot, "result.json"),
      library = path.join(sdkRoot, "lib/libzstd.a"), toolsFile = path.join(root, "tools.json");
    fs.mkdirSync(source); fs.mkdirSync(path.dirname(library), {recursive: true});
    const tools = {ar: path.join(root, "ar"), cc: path.join(root, "cc"), make: path.join(root, "make"),
      mkdir: path.join(root, "mkdir"), tar: path.join(root, "tar")};
    for (const [name, file] of Object.entries(tools)) {fs.writeFileSync(file, "fixture " + name);}
    fs.writeFileSync(toolsFile, "pinned descriptor\n"); fs.writeFileSync(library, "fixture static archive");
    const fetchedHash = /** @type {string} */ (zstdPin.sha256), toolRecords = Object.fromEntries(Object.entries(tools).map(([name, file]) => [name,
      {path: file, sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex")} ]));
    const sdkReceipt = {archive: {path: archive, sha256: fetchedHash}, config: {sha256: createHash("sha256").update(fs.readFileSync(toolsFile)).digest("hex")},
      plan: {jobs: 4}, tools: toolRecords};
    fs.writeFileSync(receipt, JSON.stringify(sdkReceipt));
    const receiptSha256 = createHash("sha256").update(fs.readFileSync(receipt)).digest("hex"), libraryHash = createHash("sha256").update(fs.readFileSync(library)).digest("hex");
    let admittedRoot = "", admittedSource = "", inputArchivePath = archive, inputArchiveHash = fetchedHash;
    const readers = {
      verifyArchive: (/** @type {string} */ path) => {
        admittedSource = path;
        return {version: 1, target: "zstd", status: "PASS", networked: true, buildOnly: true, compiled: false,
          publicationQualified: false, fetcherSha256: "f".repeat(64), recipe: {sha256: "e".repeat(64), files: {}},
          archives: {zstd: {path: archive, url: "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz",
            sha256: fetchedHash, bytes: 1}}};
      },
      sdkInputs: (/** @type {string} */ path) => {
        admittedRoot = path;
        return {archive: library, artifacts: {"lib/libzstd.a": libraryHash},
          files: {[inputArchivePath]: inputArchiveHash, [library]: libraryHash, [receipt]: receiptSha256}, receipt, receiptSha256, notices: []};
      },
    };
    const state = {source, archive, sdkRoot}, result = auditZstdSdk(state, tools, toolsFile, readers);
    assert.equal(admittedSource, source); assert.equal(admittedRoot, sdkRoot); assert.equal(result.archive, library);
    inputArchiveHash = "0".repeat(64);
    assert.throws(() => auditZstdSdk(state, tools, toolsFile, readers), /does not bind this run's pinned archive/);
    inputArchiveHash = fetchedHash; inputArchivePath = path.join(source, "other.tar.gz");
    assert.throws(() => auditZstdSdk(state, tools, toolsFile, readers), /does not bind this run's pinned archive/);
    inputArchivePath = archive;
    sdkReceipt.archive.path = path.join(source, "other.tar.gz"); fs.writeFileSync(receipt, JSON.stringify(sdkReceipt));
    assert.throws(() => auditZstdSdk(state, tools, toolsFile, readers), /does not bind this run's pinned archive/);
  });

  test.describe("State archive output safety", () => {
    test("permits a sibling archive under the base without archiving itself", {skip: linuxOnlySkip}, t => {
      const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-sibling-tar-")), source = path.join(base, "state"),
        archive = path.join(base, "state.tar"), restored = path.join(base, "restored");
      t.after(() => fs.rmSync(base, {recursive: true, force: true}));
      fs.mkdirSync(source); fs.mkdirSync(restored);
      fs.writeFileSync(path.join(source, "input.txt"), "unchanged input\n");
      assert.ok(packState(archive, base, ["state"]).bytes > 0);
      assert.ok(listState(archive, ["state"]).every(member => member === "state" || member.startsWith("state/")));
      extractState(archive, restored, ["state"]);
      assert.equal(fs.readFileSync(path.join(restored, "state/input.txt"), "utf8"), "unchanged input\n");
      assert.equal(fs.existsSync(path.join(restored, "state.tar")), false);
    });

    test("rejects selected-input overlap, aliases, traversal and existing outputs before mutation", {skip: linuxOnlySkip}, t => {
      const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-output-guard-")), source = path.join(base, "state"),
        other = path.join(base, "other"), alias = path.join(base, "alias"), existing = path.join(base, "existing.tar");
      t.after(() => fs.rmSync(base, {recursive: true, force: true}));
      fs.mkdirSync(source); fs.mkdirSync(other); fs.mkdirSync(path.join(source, "nested"));
      fs.writeFileSync(existing, "preserve me\n"); fs.symlinkSync(source, alias, "dir");
      for (const output of [path.join(source, "inside.tar"), path.join(other, "inside.tar"),
        path.join(source, "nested/../normalized.tar"), path.join(alias, "redirected.tar"), existing]) {
        assert.throws(() => packState(output, base, ["state", "other"]), /outside its inputs/);
        if (output !== existing) {assert.equal(fs.existsSync(output), false);}
      }
      const output = path.join(base, "invalid.tar");
      assert.throws(() => packState(output, base, ["../escape"]), /exact relative state paths/);
      assert.throws(() => packState(output, base, ["alias"]), /canonical regular state path/);
      assert.equal(fs.existsSync(output), false);
      assert.equal(fs.readFileSync(existing, "utf8"), "preserve me\n");
    });
  });

  test("Linux state tar round-trip preserves exact root paths, read-only receipts, modes, links and mtimes", {skip: process.platform !== "linux"}, t => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-tar-")), base = path.join(temp, "root");
    t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
    fs.mkdirSync(base);
    const source = path.join(base, "mom-native-ci-state"), ioSource = path.join(base, "mom-native-io-state"), archive = path.join(temp, "state.tar");
    fs.mkdirSync(path.join(source, "source"), {recursive: true});
    fs.mkdirSync(path.join(ioSource, "archives"), {recursive: true});
    const receipt = path.join(source, "source", "fetch.json"), executable = path.join(source, "source", "tool.sh");
    fs.writeFileSync(receipt, "immutable receipt\n", {mode: 0o600});
    fs.chmodSync(receipt, 0o444);
    fs.writeFileSync(executable, "#!/bin/sh\nexit 0\n", {mode: 0o600});
    fs.chmodSync(executable, 0o751);
    const epoch = new Date("2020-01-02T03:04:05.123Z");
    fs.utimesSync(receipt, epoch, epoch);
    fs.utimesSync(executable, epoch, epoch);
    fs.symlinkSync("tool.sh", path.join(source, "source", "tool-alias"));
    const ioArchive = path.join(ioSource, "archives", "libuv-v1.51.0.tar.gz");
    fs.writeFileSync(ioArchive, "pinned archive fixture\n", {mode: 0o444});
    fs.utimesSync(ioArchive, epoch, epoch);
    const ioArchiveBefore = {mode: fs.statSync(ioArchive).mode & 0o7777, mtime: fs.statSync(ioArchive).mtimeMs, bytes: fs.readFileSync(ioArchive)};
    const expected = {
      receipt: {mode: 0o444, mtime: fs.statSync(receipt).mtimeMs, bytes: fs.readFileSync(receipt)},
      executable: {mode: 0o751, mtime: fs.statSync(executable).mtimeMs, bytes: fs.readFileSync(executable)},
    };
    assert.deepEqual(packState(archive, base, ["mom-native-ci-state", "mom-native-io-state"]), {bytes: fs.statSync(archive).size});
    const roots = ["mom-native-ci-state", "mom-native-io-state"], members = listState(archive, roots);
    assert.equal(tarExpandedBytes(archive, members.length, 10000), tarExpandedBytes(archive, members.length));
    for (const invalid of [0, 999, NaN, Infinity, 30 * 60 * 1000 + 1]) {
      assert.throws(() => extractState(archive, base, roots, roots, invalid), /extraction timeout/);
    }
    for (const invalid of [0, 999, NaN, Infinity, 15 * 60 * 1000 + 1]) {
      assert.throws(() => tarExpandedBytes(archive, members.length, invalid), /size-listing timeout/);
    }
    for (const invalid of [0, -1, NaN, Infinity, 500001]) {
      assert.throws(() => tarExpandedBytes(path.join(base, "missing.tar"), invalid), /size-listing member count/);
    }
    assert.throws(() => listState(archive, ["mom-native-ci-state"]), /unexpected state roots/);
    fs.rmSync(source, {recursive: true});
    fs.rmSync(ioSource, {recursive: true});
    fs.mkdirSync(source); fs.writeFileSync(path.join(source, "shared-marker"), "already admitted shared state\n");
    extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"], ["mom-native-io-state"]);
    assert.deepEqual(fs.readdirSync(source), ["shared-marker"], "selected restore leaves an already-restored shared root untouched");
    assert.equal(fs.readFileSync(path.join(source, "shared-marker"), "utf8"), "already admitted shared state\n");
    assert.equal(fs.existsSync(ioSource), true);
    fs.rmSync(source, {recursive: true});
    fs.rmSync(ioSource, {recursive: true});
    assert.throws(() => extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"], ["unexpected"]), /invalid/);
    assert.throws(() => extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"], []), /invalid/);
    assert.throws(() => extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"], ["mom-native-io-state", "mom-native-io-state"]), /invalid/);
    extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"], roots, 10000);
    for (const {fileName, prior} of [
      {fileName: "fetch.json", prior: expected.receipt}, {fileName: "tool.sh", prior: expected.executable},
    ]) {
      const file = path.join(source, "source", fileName), stat = fs.statSync(file);
      assert.equal(stat.mode & 0o7777, prior.mode);
      assert.equal(stat.mtimeMs, prior.mtime);
      assert.deepEqual(fs.readFileSync(file), prior.bytes);
    }
    assert.equal(fs.readlinkSync(path.join(source, "source", "tool-alias")), "tool.sh");
    const restoredIoArchive = path.join(ioSource, "archives", "libuv-v1.51.0.tar.gz"), ioStat = fs.statSync(restoredIoArchive);
    assert.equal(ioStat.mode & 0o7777, ioArchiveBefore.mode);
    assert.equal(ioStat.mtimeMs, ioArchiveBefore.mtime);
    assert.deepEqual(fs.readFileSync(restoredIoArchive), ioArchiveBefore.bytes);
    assert.throws(() => extractState(archive, base, ["mom-native-ci-state", "mom-native-io-state"]), /existing path/);
  });

  test("NVRTC verification uses run-scoped roots and uploads receipts plus proof outputs only", {skip: linuxOnlySkip}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-nvrtc-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123456-2", runnerTemp: fs.realpathSync(runnerTemp), output: "/tmp/mom-native-llvm21-sdk-123456-2/linux-llvm21-sdk",
        env: {}}, environment = {
        MOM_NATIVE_NVRTC_LINUX_SOURCE_DIR: "/tmp/mom-native-nvrtc-linux-prerequisites-123456-2",
        MOM_NATIVE_NVRTC_LINUX_SDK_DIR: "/tmp/mom-native-nvrtc-linux-sdk-123456-2",
        MOM_NATIVE_NVRTC_WINDOWS_SOURCE_DIR: "/tmp/mom-native-nvrtc-windows-prerequisites-123456-2",
        MOM_NATIVE_NVRTC_WINDOWS_SDK_DIR: "/tmp/mom-native-nvrtc-windows-sdk-123456-2",
      }, linux = nvrtcStatePaths(paths, "linux", environment), windows = nvrtcStatePaths(paths, "windows", environment);
    assert.equal(nvrtcPins.linux.version, "13.3.33"); assert.equal(nvrtcPins.windows.version, "13.1.115");
    assert.equal(linux.source, environment.MOM_NATIVE_NVRTC_LINUX_SOURCE_DIR);
    assert.equal(linux.sdkRoot, environment.MOM_NATIVE_NVRTC_LINUX_SDK_DIR);
    assert.equal(linux.evidence, "/tmp/mom-native-nvrtc-linux-verification-123456-2");
    assert.equal(path.basename(linux.verificationTar), "native-nvrtc-linux-verification-123456-2.tar");
    assert.equal(windows.source, environment.MOM_NATIVE_NVRTC_WINDOWS_SOURCE_DIR);
    assert.equal(windows.sdkRoot, environment.MOM_NATIVE_NVRTC_WINDOWS_SDK_DIR);
    assert.equal(path.basename(windows.verificationTar), "native-nvrtc-windows-verification-123456-2.tar");
    assert.throws(() => nvrtcStatePaths(paths, "linux", {...environment,
      MOM_NATIVE_NVRTC_LINUX_SDK_DIR: "/tmp/ambient-nvrtc-sdk"}), /exactly match/);
  });

  test("NVRTC evidence packer admits only receipt and generated-code fixtures", {skip: process.platform !== "linux"}, t => {
    const outputs = ["compute_120.ptx", "compute_80.ptx", "sm_120.cubin", "sm_80.cubin"].sort(),
      proofHash = (/** @type {Buffer} */ data) => createHash("sha256").update(data).digest("hex"),
      fixture = (/** @type {"linux" | "windows"} */ platform) => {
        const workspace = fs.mkdtempSync(path.join("/var/tmp", "mom-native-ci-nvrtc-pack-")), evidenceName = path.basename(workspace) + "-evidence",
          evidence = path.join("/tmp", evidenceName), source = path.join(workspace, "source"), sdkRoot = path.join(workspace, "sdk"),
          proofDirectory = path.join(workspace, "proof"), receiptFile = path.join(workspace, "admitted-sdk.json"), verificationTar = path.join(workspace, "verification.tar"),
          state = {platform, source, sourceName: path.basename(source), sdkRoot, sdkName: path.basename(sdkRoot), evidence, evidenceName,
            proofParent: path.dirname(proofDirectory), verificationTar};
        assert.equal(fs.existsSync(evidence), false, "fixture evidence path must be new");
        fs.mkdirSync(source); fs.writeFileSync(path.join(source, "fetch.json"), JSON.stringify({target: "nvrtc-" + platform, status: "PASS"}));
        const receipt = {target: "nvrtc-sdk-" + platform, platform, status: "PASS", vendorPrebuilt: true,
          sourceRebuilt: false, compiled: false, publicationQualified: false, redistributionQualified: false};
        fs.writeFileSync(receiptFile, JSON.stringify(receipt));
        const admitted = {receipt, receiptFile, receiptSha256: "a".repeat(64)};
        t.after(() => {fs.rmSync(evidence, {recursive: true, force: true}); fs.rmSync(workspace, {recursive: true, force: true});});
        if (platform === "linux") {
          fs.mkdirSync(proofDirectory);
          const rows = outputs.map(name => {
            const data = Buffer.from(name + "\n");
            for (const directory of ["actual", "expected", "lld"]) {
              const outputDirectory = path.join(proofDirectory, directory);
              if (!fs.existsSync(outputDirectory)) {fs.mkdirSync(outputDirectory);}
              fs.writeFileSync(path.join(outputDirectory, name), data);
            }
            return {name, bytes: data.length, sha256: proofHash(data)};
          });
          const proof = {sdk: sdkRoot, vendor: {receiptSha256: admitted.receiptSha256}, files: rows,
            source: {file: "sycl/cn-gpu/cuda-expand-kernel.inc", sha256: createHash("sha256").update(fs.readFileSync(path.join(root, "sycl/cn-gpu/cuda-expand-kernel.inc"))).digest("hex")}};
          fs.writeFileSync(path.join(proofDirectory, "result.json"), JSON.stringify(proof));
          for (const name of ["build.log", "oracle-build.log", "compile.log", "oracle.log", "llvm-tools.log", "lld-build.log",
            "lld.log", "arguments.log", "dependencies.log", "syscalls.log", "isolated-paths.log"]) {
            fs.writeFileSync(path.join(proofDirectory, name), "fixture log\n");
          }
        }
        return {state, proofDirectory, admitted, workspace, receipt, outputs};
      },
      linux = fixture("linux"), linuxTar = packNvrtcVerification(linux.state, linux.proofDirectory, linux.admitted),
      linuxMembers = listState(linux.state.verificationTar, [linux.state.evidenceName]);
    assert.ok(linuxTar.bytes > 0);
    assert.ok(fs.statSync(linux.state.verificationTar).size > 0, "successful Linux evidence must create a nonempty handoff");
    assert.ok(linuxMembers.includes(linux.state.evidenceName + "/fetch.json"));
    assert.ok(linuxMembers.includes(linux.state.evidenceName + "/sdk.json"));
    for (const name of outputs) {assert.ok(linuxMembers.includes(linux.state.evidenceName + "/proof/actual/" + name));}
    assert.doesNotMatch(linuxMembers.join("\n"), /nvrtc-native|nvrtc-oracle|nvrtc-lld|libnvrtc|nvptxcompiler|\.a(?:$|\/)|\.exe/);

    const windows = fixture("windows"), windowsTar = packNvrtcVerification(windows.state, "", windows.admitted),
      windowsMembers = listState(windows.state.verificationTar, [windows.state.evidenceName]);
    assert.ok(windowsTar.bytes > 0);
    assert.deepEqual(windowsMembers.filter(name => name.endsWith(".json")).sort(), [
      windows.state.evidenceName + "/fetch.json", windows.state.evidenceName + "/sdk.json",
    ]);
    assert.doesNotMatch(windowsMembers.join("\n"), /\/proof\//, "Windows fixture exercises admission receipts only, with no compiler proof");

    const malformed = fixture("linux"), malformedFile = path.join(malformed.proofDirectory, "result.json"),
      malformedProof = JSON.parse(fs.readFileSync(malformedFile, "utf8"));
    malformedProof.files[0].name = "unexpected.ptx";
    fs.writeFileSync(malformedFile, JSON.stringify(malformedProof));
    assert.throws(() => packNvrtcVerification(malformed.state, malformed.proofDirectory, malformed.admitted), /matrix/);
    assert.equal(fs.existsSync(malformed.state.verificationTar), false, "malformed proof must not produce a handoff");

    const traversal = fixture("linux"), traversalFile = path.join(traversal.proofDirectory, "result.json"),
      traversalProof = JSON.parse(fs.readFileSync(traversalFile, "utf8")), escaped = path.join(traversal.workspace, "escape");
    traversalProof.files[0].name = "../escape";
    fs.writeFileSync(traversalFile, JSON.stringify(traversalProof));
    assert.throws(() => packNvrtcVerification(traversal.state, traversal.proofDirectory, traversal.admitted), /matrix/);
    assert.equal(fs.existsSync(escaped), false, "traversal-like proof names must not escape their output directories");
    assert.equal(fs.existsSync(traversal.state.verificationTar), false);

    const symlink = fixture("linux"), linkedFile = path.join(symlink.proofDirectory, "actual", "compute_120.ptx"),
      linkedTarget = path.join(symlink.workspace, "outside-output");
    fs.writeFileSync(linkedTarget, "outside\n"); fs.unlinkSync(linkedFile); fs.symlinkSync(linkedTarget, linkedFile);
    assert.throws(() => packNvrtcVerification(symlink.state, symlink.proofDirectory, symlink.admitted), /generated proof output changed/);
    assert.equal(fs.existsSync(symlink.state.verificationTar), false);

    const badHash = fixture("linux"), badHashFile = path.join(badHash.proofDirectory, "result.json"),
      badHashProof = JSON.parse(fs.readFileSync(badHashFile, "utf8"));
    badHashProof.files[0].sha256 = "0".repeat(64);
    fs.writeFileSync(badHashFile, JSON.stringify(badHashProof));
    assert.throws(() => packNvrtcVerification(badHash.state, badHash.proofDirectory, badHash.admitted), /generated proof output changed/);
    assert.equal(fs.existsSync(badHash.state.verificationTar), false);

    const oversized = fixture("linux"), oversizedLog = path.join(oversized.proofDirectory, "build.log");
    fs.truncateSync(oversizedLog, 32 * 1024 * 1024 + 1);
    assert.throws(() => packNvrtcVerification(oversized.state, oversized.proofDirectory, oversized.admitted), /bounded regular file/);
    assert.equal(fs.existsSync(oversized.state.verificationTar), false);
  });

  test("CUDA image paths are run-scoped and keep the public 12.6 cohort distinct", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cuda-images-paths-")), suffix = "456789-3",
      environment = {MOM_NATIVE_LINUX_SOURCES_DIR: `/tmp/mom-native-linux-source-prerequisites-${suffix}`,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: `/tmp/mom-native-dpcpp-cuda-inputs-${suffix}`,
        MOM_NATIVE_NVRTC_LINUX_SOURCE_DIR: `/tmp/mom-native-nvrtc-linux-prerequisites-${suffix}`,
        MOM_NATIVE_NVRTC_LINUX_SDK_DIR: `/tmp/mom-native-nvrtc-linux-sdk-${suffix}`,
        MOM_NATIVE_NVRTC_LINUX_126_SOURCE_DIR: `/tmp/mom-native-nvrtc-linux-12.6-prerequisites-${suffix}`,
        MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR: `/tmp/mom-native-nvrtc-linux-12.6-sdk-${suffix}`,
        MOM_NATIVE_CUDA_IMAGES_DIR: `/tmp/mom-native-cuda-images-${suffix}`},
      paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)}, state = cudaImagesStatePaths(paths, environment);
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    assert.equal(state.linux.sdkRoot, environment.MOM_NATIVE_NVRTC_LINUX_SDK_DIR);
    assert.equal(state.sdk126, environment.MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR);
    assert.equal(state.dpcpp.cudaRoot, environment.MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR + "/cuda");
    assert.equal(state.output, environment.MOM_NATIVE_CUDA_IMAGES_DIR + "/bundle");
    assert.equal(path.basename(state.header126), "cuda-12.6.inc");
    assert.equal(path.basename(state.header133), "cuda-13.3.inc");
    assert.equal(path.basename(state.tar), `native-linux-cuda-images-${suffix}.tar`);
    assert.equal(path.basename(state.noticeRoot), "notice-sources");
    assert.equal(state.noticeManifest, path.join(state.noticeRoot, "manifest.json"));
    assert.throws(() => cudaImagesStatePaths(paths, {...environment,
      MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR: "/tmp/stale-12.6-sdk"}), /exactly match/);
  });

  test("CUTLASS installer retains the pinned license and repairs only a missing cached copy", {skip: process.platform !== "linux"}, t => {
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cutlass-license-"));
    t.after(() => fs.rmSync(workspace, {recursive: true, force: true}));
    /** @param {string | Buffer} data */
    const hash = data => createHash("sha256").update(data).digest("hex"),
      license = Buffer.from("CUTLASS license fixture\n"), tensor = Buffer.from("pinned header fixture\n"),
      tree = path.join(workspace, "archive-root"), memberRoot = path.join(tree, "cutlass-4.6.1"),
      archive = path.join(workspace, "cutlass-fixture.tar.gz"), licenseSha256 = hash(license);
    fs.mkdirSync(path.join(memberRoot, "include/cute"), {recursive: true});
    fs.writeFileSync(path.join(memberRoot, "include/cute/tensor.hpp"), tensor);
    fs.writeFileSync(path.join(memberRoot, "LICENSE.txt"), license);
    const packed = spawnSync("/usr/bin/tar", ["-czf", archive, "-C", tree, "cutlass-4.6.1"], {encoding: "utf8", timeout: 5000});
    assert.equal(packed.status, 0, packed.stderr);
    const archiveSha256 = fileHash(archive);
    const sourceScript = fs.readFileSync(path.join(root, "scripts/install-cutlass.sh"), "utf8");
    assert.match(sourceScript, /MOM_CUTLASS_VERSION=v4\.6\.1/);
    assert.match(sourceScript, /MOM_CUTLASS_SHA256=455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c/);
    assert.match(sourceScript, /MOM_CUTLASS_LICENSE_SHA256=ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475/);
    const testScript = path.join(workspace, "install-cutlass.sh"), fakeBin = path.join(workspace, "bin"),
      calls = path.join(workspace, "curl-calls.log");
    fs.mkdirSync(fakeBin);
    fs.writeFileSync(testScript, sourceScript
      .replace("MOM_CUTLASS_SHA256=455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c", `MOM_CUTLASS_SHA256=${archiveSha256}`)
      .replace("MOM_CUTLASS_LICENSE_SHA256=ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475", `MOM_CUTLASS_LICENSE_SHA256=${licenseSha256}`));
    fs.writeFileSync(path.join(fakeBin, "curl"), "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = -o ]; then shift; printf 'called\\n' >>\"$TEST_CURL_CALLS\"; cp \"$TEST_CUTLASS_ARCHIVE\" \"$1\"; exit $?; fi\n  shift\ndone\nexit 91\n");
    fs.chmodSync(path.join(fakeBin, "curl"), 0o755);
    const env = {...process.env, PATH: fakeBin + path.delimiter + process.env["PATH"], TEST_CUTLASS_ARCHIVE: archive,
      TEST_CURL_CALLS: calls};
    /** @param {string} destination */
    const install = destination => spawnSync("bash", ["-ec", '. "$1"; install_cutlass_headers "$2"', "cutlass-fixture", testScript, destination],
      {encoding: "utf8", env, timeout: 10000});
    /** @param {string} destination @param {Buffer} [cachedLicense] */
    const cached = (destination, cachedLicense) => {
      fs.mkdirSync(path.join(destination, "include/cute"), {recursive: true});
      fs.writeFileSync(path.join(destination, "include/cute/tensor.hpp"), tensor);
      fs.writeFileSync(path.join(destination, ".mom-version"), archiveSha256 + "\n");
      if (cachedLicense) {fs.writeFileSync(path.join(destination, "LICENSE.txt"), cachedLicense);}
    };

    const warm = path.join(workspace, "warm-valid"); cached(warm, license);
    const warmBefore = fs.readFileSync(path.join(warm, "include/cute/tensor.hpp"));
    const warmResult = install(warm);
    assert.equal(warmResult.status, 0, warmResult.stderr);
    assert.equal(fs.existsSync(calls), false, "a valid warm cache does not need network access");
    assert.deepEqual(fs.readFileSync(path.join(warm, "include/cute/tensor.hpp")), warmBefore);

    const repaired = path.join(workspace, "warm-missing-license"); cached(repaired);
    const repairedHeader = fs.readFileSync(path.join(repaired, "include/cute/tensor.hpp")),
      repairedMarker = fs.readFileSync(path.join(repaired, ".mom-version"));
    const repairResult = install(repaired);
    assert.equal(repairResult.status, 0, repairResult.stderr);
    assert.deepEqual(fs.readFileSync(path.join(repaired, "include/cute/tensor.hpp")), repairedHeader);
    assert.deepEqual(fs.readFileSync(path.join(repaired, ".mom-version")), repairedMarker);
    assert.deepEqual(fs.readFileSync(path.join(repaired, "LICENSE.txt")), license);
    assert.equal(fs.readFileSync(calls, "utf8"), "called\n");

    fs.unlinkSync(calls);
    const tampered = path.join(workspace, "warm-tampered-license"), badLicense = Buffer.from("changed license\n"); cached(tampered, badLicense);
    const tamperResult = install(tampered);
    assert.notEqual(tamperResult.status, 0);
    assert.match(tamperResult.stderr, /LICENSE\.txt has changed/);
    assert.deepEqual(fs.readFileSync(path.join(tampered, "LICENSE.txt")), badLicense);
    assert.equal(fs.existsSync(calls), false, "tampered cached text is rejected before network access");

    const cold = path.join(workspace, "cold-install"), coldResult = install(cold);
    assert.equal(coldResult.status, 0, coldResult.stderr);
    assert.deepEqual(fs.readFileSync(path.join(cold, "LICENSE.txt")), license);
    assert.deepEqual(fs.readFileSync(path.join(cold, "include/cute/tensor.hpp")), tensor);
    assert.equal(fs.readFileSync(path.join(cold, ".mom-version"), "utf8").trim(), archiveSha256);
    assert.equal(fs.existsSync(path.join(cold, ".mom-license-sha256")), false);
  });

  test("same-run CUDA notice-source handoff binds bounded source receipts and rejects mutation", {skip: process.platform !== "linux"}, t => {
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cuda-notice-sources-"));
    t.after(() => fs.rmSync(workspace, {recursive: true, force: true}));
    /** @param {string} file @param {string | Buffer} data */
    const put = (file, data) => {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return fileHash(file);};
    /** @param {string} suffix */
    const fixture = suffix => {
      const sourceRoot = path.join(workspace, suffix, "sources"), noticeRoot = path.join(workspace, suffix, "handoff/notice-sources"),
        noticeManifest = path.join(noticeRoot, "manifest.json"), receipt = path.join(sourceRoot, "source-receipt.json"),
        receiptSha256 = put(receipt, JSON.stringify({version: 1, status: "PASS"})),
        sources = Array.from({length: 7}, (_, index) => {
          const file = `source-${index}.LICENSE.txt`, sourcePath = path.join(sourceRoot, file), body = Buffer.from(`body ${index}\n`),
            sha256 = put(sourcePath, body);
          return {component: `fixture-${index}`, version: "1", sourcePath, file, sha256, bytes: body.length,
            sourceReceipt: {path: receipt, sha256: receiptSha256}};
        });
      /** @type {CudaNoticeFixturePlan} */
      const plan = {format: "mom-cuda-notice-sources-v1", scope: "CUDA/NVRTC/CUTLASS source license bodies only", completeNativeNotices: false,
        sourceArchive: {version: "4.6.1", url: "https://example.invalid/cutlass.tar.gz", sha256: "a".repeat(64),
          member: "cutlass-4.6.1/LICENSE.txt", installer: {path: path.join(root, "scripts/install-cutlass.sh"), sha256: fileHash(path.join(root, "scripts/install-cutlass.sh"))},
          archiveMarker: {path: path.join(sourceRoot, "cutlass.marker"), sha256: put(path.join(sourceRoot, "cutlass.marker"), "archive pin\n")}},
        sources};
      fs.mkdirSync(path.dirname(noticeRoot), {recursive: true});
      return {state: {noticeRoot, noticeManifest}, plan};
    };
    const good = fixture("good"), retained = retainCudaNoticeSources(good.state, good.plan);
    assert.equal(retained.files.length, 7);
    assert.equal(retained.manifestSha256, fileHash(good.state.noticeManifest));
    assert.equal(auditCudaNoticeFiles(good.state, good.plan).manifest.completeNativeNotices, false);
    assert.deepEqual(fs.readdirSync(good.state.noticeRoot).sort(), ["manifest.json", ...good.plan.sources.map(row => row.file)].sort());
    const retainedSource = retained.files[0];
    assert.ok(retainedSource);
    fs.writeFileSync(retainedSource.path, "tampered\n");
    assert.throws(() => auditCudaNoticeFiles(good.state, good.plan), /exact body verification/);

    const missing = fixture("missing"), missingSource = missing.plan.sources[2];
    assert.ok(missingSource);
    fs.unlinkSync(missingSource.sourceReceipt.path);
    assert.throws(() => retainCudaNoticeSources(missing.state, missing.plan));
  });

  test("CUDA workers bind public SDK/license hashes and reject a 13.3-only header root", {skip: process.platform !== "linux"}, t => {
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cuda-worker-"));
    t.after(() => fs.rmSync(workspace, {recursive: true, force: true}));
    /** @param {string | Buffer} data */
    const hash = data => createHash("sha256").update(data).digest("hex");
    /** @param {string} file @param {string | Buffer} [data] */
    const put = (file, data = "fixture\n") => {
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return hash(data);
    };
    const cutlassInclude = path.join(workspace, "cutlass/include"), cudaInclude = path.join(workspace, "cuda-12.6/include"),
      compiler = path.join(workspace, "icpx"), compilerIdentity = path.join(workspace, "icpx-version.txt"),
      cutlassMarker = path.join(workspace, "cutlass/.mom-version"), cutlassLicense = path.join(workspace, "cutlass/LICENSE.txt"),
      cudaCopyrightLicenses = ["cuda-cudart-12-6", "cuda-cudart-dev-12-6", "cuda-nvml-dev-12-6"].map(name => path.join(workspace, "dpcpp/licenses", name, "copyright")),
      cudaCopyrightManifest = path.join(workspace, "dpcpp/cuda-inputs.json"), installer = path.join(root, "scripts/install-cutlass.sh");
    fs.mkdirSync(cutlassInclude, {recursive: true}); fs.mkdirSync(cudaInclude, {recursive: true});
    put(compiler); put(compilerIdentity, "Intel C++ Compiler 2026.2\n"); put(cutlassMarker, "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c\n");
    put(cutlassLicense); for (const file of cudaCopyrightLicenses) {put(file);} put(cudaCopyrightManifest);
    /** @param {string} name @param {string[]} includeRoots */
    const makeHeader = (name, includeRoots) => {
      const file = path.join(workspace, name), data = Buffer.from("fixture header pack\n");
      fs.writeFileSync(file, data);
      fs.writeFileSync(file + ".json", JSON.stringify({sourceSha256: hash(data), requiredHeaders: ["cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"],
        count: 3, bytes: 12, compressedBytes: 1, sha256: "a".repeat(64), roots: includeRoots}));
      return file;
    };
    const headerPack = makeHeader("headers-13.3.inc", [fs.realpathSync(cutlassInclude), fs.realpathSync(cudaInclude)]),
      wrongHeaderPack = makeHeader("headers-wrong.inc", [fs.realpathSync(cutlassInclude), path.join(workspace, "nvrtc-13.3/include")]);
    /** @param {"linux" | "linux-12.6"} cohort @returns {NvrtcSdkInputs} */
    const admittedFor = cohort => {
      const sdkRoot = path.join(workspace, "sdk-" + cohort), sdkJson = path.join(sdkRoot, "sdk.json"),
        sourceRoot = path.join(workspace, "sources-" + cohort), fetchJson = path.join(sourceRoot, "fetch.json"),
        tool = path.join(workspace, "tar");
      /** @type {Record<string, string>} */ const files = {};
      /** @type {NvrtcSdkInputs["receipt"]["files"]} */ const outputs = {};
      fs.mkdirSync(sdkRoot, {recursive: true}); fs.mkdirSync(sourceRoot, {recursive: true});
      const toolSha256 = put(tool, "pinned tar tool fixture\n"); files[tool] = toolSha256;
      const licenseNames = ["licenses/nvrtc/LICENSE", ...(cohort === "linux" ? ["licenses/nvptxcompiler/LICENSE"] : [])];
      for (const name of [...licenseNames, "include/nvrtc.h", "lib64/libnvrtc.so.12"]) {
        const file = path.join(sdkRoot, name), data = name + "\n", digest = put(file, data),
          archive = name.startsWith("licenses/nvptxcompiler/") ? "nvptxcompiler" : "nvrtc";
        files[file] = digest;
        outputs[name] = {kind: "file", bytes: Buffer.byteLength(data), sha256: digest, source: {archive, member: name}};
      }
      /** @type {NvrtcSdkInputs["receipt"]["source"]["receipt"]["archives"]} */ const archives = {};
      /** @type {Array<"nvrtc" | "nvptxcompiler">} */ const archiveNames = cohort === "linux" ? ["nvrtc", "nvptxcompiler"] : ["nvrtc"];
      for (const archive of archiveNames) {
        const archivePath = path.join(sourceRoot, `cuda_${archive}-${cohort}.tar.xz`), archiveSha256 = put(archivePath, "pinned archive fixture\n");
        files[archivePath] = archiveSha256;
        archives[archive] = {path: archivePath, url: `https://example.invalid/${archive}.tar.xz`, sha256: archiveSha256, bytes: 1};
      }
      const vendorVersion = cohort === "linux-12.6" ? nvrtcPins["linux-12.6"].version : nvrtcPins.linux.version;
      /** @type {NvrtcSdkInputs["receipt"]["source"]["receipt"]} */
      const fetchReceipt = {version: 1, target: "nvrtc-" + cohort, status: "PASS", networked: true, buildOnly: true,
        compiled: false, publicationQualified: false, fetcherSha256: "d".repeat(64), recipe: {sha256: "e".repeat(64), files: {}}, archives};
      const fetchSha256 = put(fetchJson, JSON.stringify(fetchReceipt)); files[fetchJson] = fetchSha256;
      /** @type {NvrtcSdkInputs["receipt"]} */ const receipt = {version: 1, target: "nvrtc-sdk-linux",
        platform: "linux", vendorVersion, status: "PASS", networked: false, buildOnly: true, compiled: false,
        sourceRebuilt: false, vendorPrebuilt: true, publicationQualified: false, redistributionQualified: false, cohort,
        producerSha256: "b".repeat(64), recipe: {sha256: "c".repeat(64), files: {}},
        tool: {path: tool, sha256: toolSha256, version: "tar fixture"},
        source: {directory: sourceRoot, receipt: fetchReceipt, receiptSha256: fetchSha256}, output: sdkRoot, files: outputs};
      const receiptSha256 = put(sdkJson, JSON.stringify(receipt)); files[sdkJson] = receiptSha256;
      return {receipt, receiptFile: sdkJson, receiptSha256, files};
    };
    const common = {cudaInclude, cutlassInclude, cutlassMarker, cutlassLicense, cudaCopyrightLicenses,
      cudaCopyrightManifest, compiler, compilerIdentity, installer};
    const baseline = cudaImageWorkerRecipe(admittedFor("linux-12.6"), {...common, cohort: "linux-12.6", headerPack});
    assert.ok(baseline.worker.inputs[path.join(workspace, "sdk-linux-12.6/licenses/nvrtc/LICENSE")]);
    assert.ok(baseline.worker.args.includes("-lnvrtc"));
    assert.equal(baseline.worker.environment.LD_LIBRARY_PATH, path.join(workspace, "sdk-linux-12.6/lib64"));
    const modern = cudaImageWorkerRecipe(admittedFor("linux"), {...common, cohort: "linux", headerPack});
    assert.ok(modern.worker.inputs[path.join(workspace, "sdk-linux/licenses/nvptxcompiler/LICENSE")]);
    assert.ok(modern.worker.inputs[cutlassLicense]);
    for (const file of cudaCopyrightLicenses) {assert.ok(modern.worker.inputs[file]);}
    assert.ok(modern.worker.inputs[cudaCopyrightManifest]);
    assert.ok(modern.worker.inputs[fs.realpathSync(installer)]);
    assert.throws(() => cudaImageWorkerRecipe(admittedFor("linux-12.6"), {...common, cohort: "linux", headerPack}), /exact admitted Linux NVRTC cohort/);
    assert.throws(() => cudaImageWorkerRecipe(admittedFor("linux"), {...common, cohort: "linux", headerPack: wrongHeaderPack}), /same-run CUDA 12\.6/);
    const withoutLicense = admittedFor("linux"), missing = path.join(withoutLicense.receipt.output, "licenses/nvptxcompiler/LICENSE");
    delete withoutLicense.files[missing];
    assert.throws(() => cudaImageWorkerRecipe(withoutLicense, {...common, cohort: "linux", headerPack}), /every admitted NVRTC license hash/);

    // Real pinned license bytes exercise the entire source-plan -> image-recipe -> handoff
    // audit; the generic transport fixture cannot catch mismatched record projections.
    put(cutlassLicense, fs.readFileSync(path.join(root, "tests/fixtures/cutlass-LICENSE.txt")));
    const baselineSdk = admittedFor("linux-12.6"), modernSdk = admittedFor("linux"),
      packageVersions = {"cuda-cudart-12-6": "12.6.85-1", "cuda-cudart-dev-12-6": "12.6.85-1", "cuda-nvml-dev-12-6": "12.6.77-1"},
      licenses = Object.entries(packageVersions).map(([name, version]) => {
        const file = path.join(workspace, "dpcpp/licenses", name, "copyright");
        return {package: name, version, path: file, sha256: fileHash(file), sourcePath: "/usr/share/doc/" + name + "/copyright"};
      }), dpcpp = {cuda: {packages: packageVersions, licenses}}, noticeRoot = path.join(workspace, "handoff/notice-sources"),
      state = {dpcpp: {root: path.join(workspace, "dpcpp"), manifest: cudaCopyrightManifest},
        sdk126: baselineSdk.receipt.output, linux: {sdkRoot: modernSdk.receipt.output}, cutlassMarker, cutlassLicense,
        noticeRoot, noticeManifest: path.join(noticeRoot, "manifest.json")},
      plan = cudaNoticeSourcePlan(state, dpcpp, baselineSdk, modernSdk),
      compilers = [{major: 12, minor: 6, inputs: cudaImageWorkerRecipe(baselineSdk, {...common, cohort: "linux-12.6", headerPack}).worker.inputs},
        {major: 13, minor: 3, inputs: cudaImageWorkerRecipe(modernSdk, {...common, cohort: "linux", headerPack}).worker.inputs}];
    fs.mkdirSync(path.dirname(noticeRoot)); retainCudaNoticeSources(state, plan);
    assert.equal(auditCudaNoticeSources(state, dpcpp, {compilers}).files.length, 7);
    const modernCompiler = compilers[1], nvml = licenses[2]; assert.ok(modernCompiler && nvml);
    const originalInputs = {...modernCompiler.inputs};
    delete modernCompiler.inputs[nvml.path];
    assert.throws(() => auditCudaNoticeSources(state, dpcpp, {compilers}), /omits notice source/);
    modernCompiler.inputs = originalInputs;
    const retainedNvml = path.join(noticeRoot, "cuda-nvml-dev-12.6-copyright.txt");
    fs.writeFileSync(retainedNvml, "modified NVML terms\n");
    assert.throws(() => auditCudaNoticeSources(state, dpcpp, {compilers}), /exact body verification/);
  });

  test("rejects traversal-like tar paths before extraction", () => {
    for (const name of ["../escape", "/absolute", "a/../b", "a\\b", "a\nb"]) {assert.equal(safeRelative(name), false);}
    assert.equal(safeRelative("mom-native-state/source/fetch.json"), true);
  });
});

test.describe("Linux source-JIT prerequisite handoffs", () => {
  const tools = {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja",
    cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"};
  /** @param {string} tag */
  const fixture = tag => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-sdk-" + tag + "-")),
      suffix = "987654-" + (tag === "plan" ? "1" : "2"), paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp)},
      environment = {MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix,
        MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/mom-native-gnu-headers-" + suffix,
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-" + suffix,
        MOM_NATIVE_LINUX_JIT_SDK_DIR: "/tmp/mom-native-linux-jit-sdk-" + suffix},
      state = linuxJitSdkStatePaths(paths, environment);
    return {runnerTemp, paths, environment, state};
  };

  test("uses the exact same-run roots and distinct mutable compiler-state artifacts", {skip: linuxOnlySkip}, t => {
    const f = fixture("plan"); t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    assert.equal(f.state.root, f.environment.MOM_NATIVE_LINUX_JIT_SDK_DIR);
    assert.equal(f.state.output, path.join(f.state.root, "compiler-sdk"));
    assert.equal(path.basename(f.state.phaseTar("configure")), "native-linux-jit-sdk-987654-1-configured.tar");
    assert.equal(path.basename(f.state.phaseTar("link")), "native-linux-jit-sdk-987654-1-link.tar");
    assert.equal(path.basename(f.state.inputTars[0]?.file || ""), "native-linux-sdk-state-987654-1-sources.tar");
    assert.equal(path.basename(f.state.inputTars[1]?.file || ""), "native-linux-dpcpp-cuda-inputs-987654-1.tar");
    assert.equal(path.basename(f.state.inputTars[2]?.file || ""), "native-linux-gnu-headers-987654-1.tar");
    assert.equal(path.basename(f.state.inputTars[3]?.file || ""), "native-linux-jit-resources-987654-1.tar");
    assert.throws(() => linuxJitSdkStatePaths(f.paths, {...f.environment, MOM_NATIVE_LINUX_JIT_SDK_DIR: "/tmp/other"}), /exactly match/);
  });

  test("all five producer phases use the existing phase API and chain only the prior build tar", {skip: linuxOnlySkip}, t => {
    const f = fixture("plan-phases"); t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    for (const [index, phase] of ["configure", "support", "targets", "frontend", "link"].entries()) {
      const plan = linuxJitSdkBuildPlan(f.state, phase || "", tools);
      assert.equal(plan.previous, index ? ["configure", "support", "targets", "frontend"][index - 1] : "");
      assert.equal(plan.commands[0]?.script, "scripts/build-native-jit-sdk.js");
      assert.deepEqual(plan.commands[0]?.args, [f.state.configFile, f.state.output, phase]);
      assert.equal(path.basename(plan.artifact.file), "native-linux-jit-sdk-987654-2-" + (phase === "configure" ? "configured" : phase) + ".tar");
      assert.equal(plan.inputTars.length, 4, "same immutable inputs are restored without being repackaged");
      if (phase === "configure") {
        assert.ok(plan.config);
        assert.equal(plan.config.sources, f.state.linux.compilerPreparedReceipt);
        assert.equal(plan.config.sdk, f.state.dpcpp.dpcppRoot);
        assert.equal(plan.config.resources, path.join(f.state.resources.root, "resources"));
        assert.deepEqual(plan.config.tools, tools); assert.equal(plan.config.jobs, 4);
      } else {
        assert.equal(path.basename(plan.previousTar), "native-linux-jit-sdk-987654-2-" + plan.previousArtifact + ".tar");
      }
    }
    for (const phase of ["all", "install", "isolation"]) {assert.throws(() => linuxJitSdkBuildPlan(f.state, phase, tools), /existing producer phase/);}
    assert.equal(linuxJitSdkArtifactPhase("configure"), "configured");
  });

  test("input manifest must agree with the resource receipt and all four original archive hashes", {skip: linuxOnlySkip}, t => {
    const f = fixture("manifest"); t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    /** @type {Record<string,string>} */ const hashes = Object.fromEntries(f.state.inputTars.map((row, index) =>
      [row.file, String(index + 1).repeat(64)]));
    const sourceRows = f.state.inputTars.slice(0, 3).map(row => ({role: row.role, name: path.basename(row.file), root: row.root, sha256: hashes[row.file] || ""})),
      resourceRecord = {inputArchives: sourceRows}, manifest = linuxJitSdkInputManifest(f.state, resourceRecord, hashes);
    assert.equal(manifest.version, 1);
    assert.equal(manifest.run, "987654-2");
    assert.deepEqual(manifest.archives.map(row => row.role), ["linux-sources", "dpcpp-cuda-inputs", "gnu-headers", "jit-resources"]);
    assert.throws(() => linuxJitSdkInputManifest(f.state, {inputArchives: sourceRows.slice(1)}, hashes), /three immutable/);
    const substituted = structuredClone(sourceRows);
    if (substituted[1]) {substituted[1].sha256 = "f".repeat(64);}
    assert.throws(() => linuxJitSdkInputManifest(f.state, {inputArchives: substituted}, hashes), /three immutable/);
    assert.throws(() => linuxJitSdkInputManifest(f.state, {inputArchives: sourceRows}, {...hashes, extra: "a".repeat(64)}), /archive proof/);
  });

  test("source-JIT restore rejects a wrong external DPC++ bundle pin before extracting any input", {skip: linuxOnlySkip}, t => {
    const suffix = `restore-pin-${process.pid}-${Date.now()}`, runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-restore-pin-")),
      environment = {
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: "/tmp/mom-native-dpcpp-cuda-inputs-" + suffix,
        MOM_NATIVE_GNU_HEADERS_DIR: "/tmp/mom-native-gnu-headers-" + suffix,
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-" + suffix,
        MOM_NATIVE_LINUX_JIT_SDK_DIR: "/tmp/mom-native-linux-jit-sdk-" + suffix,
        MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256: "a".repeat(64),
        MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256: "b".repeat(64),
      }, paths = /** @type {any} */ ({suffix, runnerTemp: fs.realpathSync(runnerTemp), env: environment});
    const state = linuxJitSdkStatePaths(paths, environment), incoming = path.join(paths.runnerTemp, "incoming");
    assert.ok(![state.resources.root, state.dpcpp.root, state.linux.source, state.gnu.root, state.root].some(fs.existsSync));
    t.after(() => {
      fs.rmSync(state.resources.root, {recursive: true, force: true});
      fs.rmSync(state.dpcpp.root, {recursive: true, force: true});
      fs.rmSync(state.linux.source, {recursive: true, force: true});
      fs.rmSync(state.gnu.root, {recursive: true, force: true});
      fs.rmSync(state.root, {recursive: true, force: true});
      fs.rmSync(runnerTemp, {recursive: true, force: true});
    });
    fs.mkdirSync(incoming);
    fs.mkdirSync(state.resources.resources, {recursive: true, mode: 0o700});
    fs.writeFileSync(path.join(state.resources.resources, "fixture.txt"), "valid resource root\n");
    packState(state.inputTars[3]?.file || "", "/tmp", [state.resources.name]);
    fs.rmSync(state.resources.root, {recursive: true});
    for (const row of state.inputTars.slice(0, 3)) {fs.writeFileSync(row.file, "not the pinned DPC++/CUDA bundle\n");}
    assert.throws(() => restoreLinuxJitSdkInputs(paths, state, "configure"), /externally pinned same-run source tar/);
    assert.equal(fs.existsSync(state.resources.root), false, "the valid resource tar must not be extracted before the bundle pin is checked");
    assert.equal(fs.existsSync(state.linux.source), false);
    assert.equal(fs.existsSync(state.dpcpp.root), false);
    assert.equal(fs.existsSync(state.gnu.root), false);
  });

  test("configure receipt hashes bind the actual tool manifest and command log", {skip: linuxOnlySkip}, t => {
    const f = fixture("configure-hashes"), output = path.join(f.runnerTemp, "compiler-sdk");
    t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true})); fs.mkdirSync(output);
    const digest = (/** @type {string} */ name) => createHash("sha256").update(fs.readFileSync(path.join(output, name))).digest("hex");
    const contents = {"configured-tools.json": "{}\n", "commands.log": "original compiler commands\n"};
    for (const [name, text] of Object.entries(contents)) {fs.writeFileSync(path.join(output, name), text);}
    const receipt = {configuredSha256: digest("configured-tools.json"), commandsSha256: digest("commands.log")};
    assert.doesNotThrow(() => auditLinuxJitSdkConfigureArtifacts(output, receipt));
    for (const [name, text] of Object.entries(contents)) {
      fs.appendFileSync(path.join(output, name), "changed\n");
      assert.throws(() => auditLinuxJitSdkConfigureArtifacts(output, receipt), new RegExp(name.replace(".", "\\.")));
      fs.writeFileSync(path.join(output, name), text);
    }
  });

  test("bootstrap overlay maps real GNU roots and the pinned OpenCL CL directory", {skip: linuxOnlySkip}, t => {
    const f = fixture("overlay-plan"); t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    const gnuRoot = "/tmp/gnu-run/include-root", opencl = {root: "/tmp/source-run/opencl", revision: "a".repeat(40)},
      plan = linuxJitBootstrapOverlayPlan(f.state.resources, {headers: {root: gnuRoot, files: []}, notices: []}, opencl);
    assert.deepEqual(plan.map(row => row.virtual), ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target", "include/mom-system", "include/CL"]);
    assert.deepEqual(plan.map(row => row.source), [
      path.join(gnuRoot, "include/mom-cxx"), path.join(gnuRoot, "include/mom-cxx-target"),
      path.join(gnuRoot, "include/mom-system-target"), path.join(gnuRoot, "include/mom-system"), path.join(opencl.root, "CL"),
    ]);
    assert.deepEqual(plan.map(row => row.link), plan.map(row => path.join(f.state.resources.bootstrapOverlay, ...row.virtual.split("/"))));
    assert.throws(() => linuxJitBootstrapOverlayPlan(f.state.resources, {headers: {root: gnuRoot, files: []}, notices: []},
      {...opencl, revision: "bad"}), /pinned OpenCL-Headers/);
  });

  test("bootstrap overlay links real headers and rejects missing or replaced roots", {skip: linuxOnlySkip}, t => {
    const f = fixture("overlay-real"), gnuRoot = path.join(f.runnerTemp, "gnu"), openclRoot = path.join(f.runnerTemp, "opencl"),
      compilerReceipt = path.join(f.runnerTemp, "compiler-sources.json");
    t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    const roots = ["mom-cxx", "mom-cxx-target", "mom-system-target", "mom-system"];
    for (const root of roots) {const dir = path.join(gnuRoot, "include", root); fs.mkdirSync(dir, {recursive: true}); fs.writeFileSync(path.join(dir, "real.h"), "header\n");}
    fs.mkdirSync(path.join(openclRoot, "CL"), {recursive: true}); fs.writeFileSync(path.join(openclRoot, "CL", "cl.h"), "OpenCL-Headers\n");
    fs.writeFileSync(path.join(gnuRoot, "result.json"), "{}\n"); fs.writeFileSync(compilerReceipt, "{}\n");
    const state = f.state.resources;
    state.bootstrapOverlay = path.join(f.runnerTemp, "overlay"); state.bootstrapOverlayEvidence = path.join(f.runnerTemp, "overlay.json");
    state.gnu.root = gnuRoot; state.linux.compilerPreparedReceipt = compilerReceipt;
    const admitted = {headers: {root: gnuRoot, files: []}, notices: []}, opencl = {root: openclRoot, revision: "b".repeat(40)},
      receipt = seedLinuxJitBootstrapOverlay(state, admitted, opencl);
    assert.equal(auditLinuxJitBootstrapOverlay(state, receipt).roots.length, 5);
    assert.equal(fs.readFileSync(path.join(state.bootstrapOverlay, "include/CL/cl.h"), "utf8"), "OpenCL-Headers\n");
    assert.equal(fs.readlinkSync(path.join(state.bootstrapOverlay, "include/mom-cxx")), path.join(gnuRoot, "include/mom-cxx"));
    fs.unlinkSync(path.join(state.bootstrapOverlay, "include/CL")); fs.symlinkSync(gnuRoot, path.join(state.bootstrapOverlay, "include/CL"), "dir");
    assert.throws(() => auditLinuxJitBootstrapOverlay(state, receipt), /link changed/);
    const missing = {...state, bootstrapOverlay: path.join(f.runnerTemp, "missing-overlay"),
      bootstrapOverlayEvidence: path.join(f.runnerTemp, "missing-overlay.json")};
    fs.unlinkSync(path.join(openclRoot, "CL", "cl.h"));
    assert.throws(() => seedLinuxJitBootstrapOverlay(missing, admitted, opencl), /source is missing/);
  });

  test("static DPC++ continuation uses the existing isolation and SDK producers", {skip: linuxOnlySkip}, t => {
    const f = fixture("static-plan"); t.after(() => fs.rmSync(f.runnerTemp, {recursive: true, force: true}));
    const plan = linuxJitSdkBuildPlan(f.state, "static-dpcpp", tools);
    assert.equal(plan.previous, "link");
    assert.equal(path.basename(plan.previousTar), "native-linux-jit-sdk-987654-2-link.tar");
    assert.deepEqual(plan.commands[0], {script: "scripts/native-compiler-isolation.js",
      args: [path.join(f.state.output, "build"), f.state.isolation]});
    assert.equal(plan.commands[1]?.script, "scripts/build-native-dpcpp-sdk.js");
    assert.deepEqual(plan.commands[1]?.args, [f.state.linux.preparedReceipt, f.state.dpcpp.dpcppRoot, f.state.dpcpp.cudaRoot,
      f.state.isolatedObject, f.state.staticOutput, tools.cmake, tools.ninja, "4", tools.cc, tools.cxx]);
    assert.equal(path.basename(plan.artifact.file), "native-linux-jit-sdk-987654-2-static-dpcpp.tar");
  });

  test("proof-only runtime handoff preserves original bytes and rejects substituted archives", {skip: linuxOnlySkip}, t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-runtime-proof-"));
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    const source = path.join(base, "source"), packageBase = path.join(base, "package-base"), runnerTemp = path.join(base, "runner-temp");
    fs.mkdirSync(source); fs.mkdirSync(packageBase); fs.mkdirSync(runnerTemp);
    const proof = path.join(source, "native-dpcpp-jit"), receipt = path.join(source, "result.json"),
      digest = (/** @type {Buffer | string} */ data) => createHash("sha256").update(data).digest("hex");
    fs.writeFileSync(proof, "source jit runtime fixture\n", {mode: 0o755}); fs.chmodSync(proof, 0o755);
    const result = {version: 1, kind: "original-source-jit-sdk", target: "linux", proof: {path: proof, sha256: digest(fs.readFileSync(proof))},
      compiled: true, buildOnly: true, executed: false, publicationQualified: false};
    fs.writeFileSync(receipt, JSON.stringify(result) + "\n", {mode: 0o600});
    const transfer = linuxJitRuntimeProofPaths({suffix: "123456-2", runnerTemp}), name = transfer.name,
      packed = packLinuxJitRuntimeProofFiles(proof, receipt, packageBase, name, transfer.archive), restore = path.join(base, "restore");
    assert.equal(path.basename(transfer.archive), "native-linux-jit-runtime-proof-123456-2.tar");
    fs.mkdirSync(restore);
    const admitted = extractLinuxJitRuntimeProof(transfer.archive, restore, name, proof,
      {proofSha256: packed.proofSha256, resultSha256: packed.resultSha256, bundleSha256: packed.bundleSha256, bundleBytes: String(packed.bundleBytes)});
    assert.deepEqual(fs.readdirSync(admitted.root).sort(), ["native-dpcpp-jit", "result.json"]);
    assert.equal(admitted.proofSha256, digest(fs.readFileSync(proof)));
    assert.equal(admitted.resultSha256, digest(fs.readFileSync(receipt)));
    assert.equal(fs.statSync(admitted.proof).mode & 0o111, 0o111);
    assert.equal(fs.readFileSync(admitted.receipt, "utf8"), fs.readFileSync(receipt, "utf8"));
    assert.throws(() => extractLinuxJitRuntimeProof(transfer.archive, path.join(base, "no-output"), name, proof,
      {proofSha256: "", resultSha256: packed.resultSha256, bundleSha256: packed.bundleSha256, bundleBytes: String(packed.bundleBytes)}), /expectations are invalid/);

    const missingRootName = "missing-proof-result-123456-2", missingRoot = path.join(packageBase, missingRootName),
      missingArchive = path.join(runnerTemp, "missing.tar");
    fs.mkdirSync(missingRoot); fs.copyFileSync(proof, path.join(missingRoot, "native-dpcpp-jit"));
    packState(missingArchive, packageBase, [missingRootName]);
    assert.throws(() => extractLinuxJitRuntimeProof(missingArchive, path.join(base, "missing-restore"), missingRootName, proof,
      {proofSha256: packed.proofSha256, resultSha256: packed.resultSha256, bundleSha256: digest(fs.readFileSync(missingArchive)),
        bundleBytes: String(fs.statSync(missingArchive).size)}), /missing or has unexpected members/);

    fs.appendFileSync(transfer.archive, "tampered");
    assert.throws(() => extractLinuxJitRuntimeProof(transfer.archive, path.join(base, "tampered"), name, proof,
      {proofSha256: packed.proofSha256, resultSha256: packed.resultSha256, bundleSha256: packed.bundleSha256, bundleBytes: String(packed.bundleBytes)}), /bundle bytes or hash changed/);
    assert.throws(() => packLinuxJitRuntimeProofFiles(path.join(source, "missing"), receipt, packageBase, name, path.join(runnerTemp, "missing-output.tar")),
      /regular executable\/file/);
    const changed = {...result, executed: true}, changedReceipt = path.join(source, "changed-result.json");
    fs.writeFileSync(changedReceipt, JSON.stringify(changed));
    assert.throws(() => packLinuxJitRuntimeProofFiles(proof, changedReceipt, packageBase, name, path.join(runnerTemp, "changed.tar")), /build-only receipt/);
  });

  test("runtime proof mount validation accepts normal paths and rejects per-path delimiters", {skip: process.platform !== "linux"}, () => {
    const run = (/** @type {string[]} */ paths) => spawnSync("bash", ["-c", 'source "$1" && shift && validate_mount_paths "$@"', "bash", runtimeScript, ...paths],
        {env: {PATH: process.env["PATH"]}, encoding: "utf8", timeout: 5000}),
      normal = ["/tools/node", "/checkout/tests/native-jit-runtime.js", "/tmp/runtime proof/native-dpcpp-jit"];
    assert.equal(run(normal).status, 0, "ordinary paths must not be rejected by joining them with a forbidden delimiter");
    for (const index of normal.keys()) {
      for (const suffix of [",extra", "\nextra", "\rextra"]) {
        assert.equal(run(normal.map((value, position) => position === index ? "/tmp/proof" + suffix : value)).status, 2);
      }
    }
  });

  test("runtime proof dispatch passes the complete container command to its timeout", {skip: process.platform !== "linux"}, t => {
    const evidence = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-dispatch-"));
    t.after(() => fs.rmSync(evidence, {recursive: true, force: true}));
    const mock = `timeout() {
      test "$#" -eq 13 && test "$8" = --env &&
      test "$9" = MOM_NATIVE_JIT_RUNTIME_FIXTURE=/proof/native-dpcpp-jit &&
      test "\${10}" = proof-fixture && test "\${11}" = /bin/bash && test "\${12}" = -ec || return 91
      [[ "\${13}" == *'/qa/tests/native-jit-runtime.js'* ]] || return 92
      [[ "\${13}" == *'# tests 11'* && "\${13}" == *'sm_120 meowpow'* ]] || return 93
      printf 'complete runtime dispatch\\n'
    }`;
    /** Source the runner, replace `timeout` with a mock, and check the recorded dispatch status. @param {string} selected @param {number} status */
    const dispatch = (selected, status) => spawnSync("bash", ["-c", 'source "$1"\n' + selected + '\ndispatch_runtime\ntest "$exec_status" -eq ' + status,
      "bash", runtimeScript], {env: {PATH: process.env["PATH"], evidence, name: "proof-fixture"}, encoding: "utf8", timeout: 5000});
    const result = dispatch(mock, 0);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.equal(fs.readFileSync(path.join(evidence, "container.log"), "utf8"), "complete runtime dispatch\n");
    const expired = dispatch("timeout() { printf 'partial\\n'; return 124; }", 124);
    assert.equal(expired.status, 0, "a timed-out dispatch records the timeout status instead of aborting the evidence path");
    assert.equal(fs.readFileSync(path.join(evidence, "container.log"), "utf8"), "partial\n");
  });

  test("runtime proof failure evidence is bundled only when the evidence directory exists", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-jit-evidence-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const env = {PATH: process.env["PATH"], RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: "123456-2"},
      bundle = path.join(runnerTemp, "native-linux-jit-runtime-failure-123456-2.tar"),
      run = (/** @type {string} */ command) => spawnSync("bash", [runtimeScript, command], {env, encoding: "utf8", timeout: 10000});
    assert.equal(run("bundle-evidence").status, 0);
    assert.equal(fs.existsSync(bundle), false, "no bundle is created before the run step creates evidence");
    fs.mkdirSync(path.join(runnerTemp, "native-linux-jit-runtime-123456-2"));
    fs.writeFileSync(path.join(runnerTemp, "native-linux-jit-runtime-123456-2", "host.log"), "host log\n");
    assert.equal(run("bundle-evidence").status, 0);
    assert.deepEqual(listState(bundle, ["native-linux-jit-runtime-123456-2"]).map(name => name.replace(/\/$/, "")).sort(),
      ["native-linux-jit-runtime-123456-2", "native-linux-jit-runtime-123456-2/host.log"]);
    const usage = run("unknown");
    assert.equal(usage.status, 2);
    assert.match(usage.stderr, /^\[native-ci\] Usage/);
  });

  test("Linux descriptor preserves the exact five profiles and current-run source/link closure", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-linux-descriptor-paths-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "123456-2", tmpRoot = (/** @type {string} */ name) => "/tmp/" + name + "-" + suffix,
      paths = {suffix, runnerTemp: fs.realpathSync(runnerTemp), source: tmpRoot("mom-native-prerequisites"),
        sourceName: "mom-native-prerequisites-" + suffix, sdkRoot: tmpRoot("mom-native-llvm21-sdk"),
        sdkName: "mom-native-llvm21-sdk-" + suffix, output: tmpRoot("mom-native-llvm21-sdk") + "/linux-llvm21-sdk",
        ioSource: tmpRoot("mom-native-io-prerequisites"), ioSourceName: "mom-native-io-prerequisites-" + suffix,
        ioSdkRoot: tmpRoot("mom-native-io-openssl-sdk"), ioSdkName: "mom-native-io-openssl-sdk-" + suffix,
        ioSdk: tmpRoot("mom-native-io-openssl-sdk") + "/io-sdk", opensslSdk: tmpRoot("mom-native-io-openssl-sdk") + "/openssl-sdk",
        opensslTools: tmpRoot("mom-native-io-openssl-sdk") + "/openssl-tools.json", env: process.env,
        zstdSource: tmpRoot("mom-native-zstd-prerequisites"), zstdSourceName: "mom-native-zstd-prerequisites-" + suffix,
        zstdSdkRoot: tmpRoot("mom-native-zstd-sdk"), zstdSdkName: "mom-native-zstd-sdk-" + suffix,
        linuxSource: tmpRoot("mom-native-linux-source-prerequisites"), linuxSourceName: "mom-native-linux-source-prerequisites-" + suffix,
        linuxJitResourcesRoot: tmpRoot("mom-native-linux-jit-resources"), linuxJitResourcesName: "mom-native-linux-jit-resources-" + suffix,
        rocmComgrSdkRoot: tmpRoot("mom-native-rocm-comgr-sdk"), rocmResourcesRoot: tmpRoot("mom-native-rocm-resources"),
        rocmHipHeadersRoot: tmpRoot("mom-native-rocm-hip-headers"), rocmHostSdkRoot: tmpRoot("mom-native-rocm-host-sdk"),
        linuxSdkDescriptorRoot: tmpRoot("mom-native-linux-sdk-descriptor"), linuxSdkDescriptorName: "mom-native-linux-sdk-descriptor-" + suffix},
      environment = {
        MOM_NATIVE_LINUX_SOURCES_DIR: paths.linuxSource, MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: tmpRoot("mom-native-dpcpp-cuda-inputs"),
        MOM_NATIVE_NVRTC_LINUX_SOURCE_DIR: tmpRoot("mom-native-nvrtc-linux-prerequisites"), MOM_NATIVE_NVRTC_LINUX_SDK_DIR: tmpRoot("mom-native-nvrtc-linux-sdk"),
        MOM_NATIVE_NVRTC_LINUX_126_SOURCE_DIR: tmpRoot("mom-native-nvrtc-linux-12.6-prerequisites"),
        MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR: tmpRoot("mom-native-nvrtc-linux-12.6-sdk"), MOM_NATIVE_CUDA_IMAGES_DIR: tmpRoot("mom-native-cuda-images"),
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: paths.linuxJitResourcesRoot, MOM_NATIVE_LINUX_JIT_SDK_DIR: tmpRoot("mom-native-linux-jit-sdk"),
        MOM_NATIVE_GNU_HEADERS_DIR: tmpRoot("mom-native-gnu-headers"), MOM_NATIVE_ACPP_SOURCE_DIR: tmpRoot("mom-native-acpp-prerequisites"),
        MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR: tmpRoot("mom-native-acpp-cuda-hip-sdk"),
        MOM_NATIVE_HERMES_SOURCE_DIR: tmpRoot("mom-native-hermes-prerequisites"), MOM_NATIVE_HERMES_SDK_DIR: tmpRoot("mom-native-hermes-sdk"),
        MOM_NATIVE_OPENMP_SDK_DIR: tmpRoot("mom-native-openmp-sdk"), MOM_NATIVE_ROCM_SOURCE_DIR: tmpRoot("mom-native-rocm-prerequisites"),
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: paths.rocmComgrSdkRoot, MOM_NATIVE_ROCM_RESOURCES_DIR: paths.rocmResourcesRoot,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: paths.rocmHipHeadersRoot, MOM_NATIVE_ROCM_HOST_SDK_DIR: paths.rocmHostSdkRoot,
        MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: paths.linuxSdkDescriptorRoot,
      }, state = linuxNativeSdkDescriptorStatePaths(paths, environment), tools = {
        cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx",
      }, linker = path.join(runnerTemp, "intel-linker-fixture", "ld.lld"), config = linuxNativeSdkDescriptorConfig(state, tools, linker);
    assert.equal(config.cpu.env.CXX, tools.cxx, "Core uses the same explicitly selected Intel compiler as the SDK");
    assert.deepEqual(state.inputRows.map(row => row.artifact), [
      "native-linux-sdk-state-123456-2-sources.tar", "native-linux-dpcpp-cuda-inputs-123456-2.tar",
      "native-linux-gnu-headers-123456-2.tar", "native-linux-cuda-images-123456-2.tar", "native-linux-jit-sdk-123456-2-static-dpcpp.tar",
      "native-acpp-state-123456-2-sources.tar", "native-linux-acpp-cuda-hip-sdk-123456-2.tar",
      "native-linux-hermes-sdk-123456-2.tar", "native-linux-rocm-host-sdk-123456-2.tar",
      "native-linux-openmp-sdk-123456-2.tar", "native-linux-zstd-sdk-123456-2.tar", "native-linux-io-openssl-sdk-123456-2.tar",
    ]);
    assert.equal(state.rocm.llvmSdkRoot, paths.sdkRoot);
    assert.equal(state.rocm.llvmSdkName, paths.sdkName);
    assert.equal(state.llvmSdk, state.rocm.llvmSdk, "the ROCm host tar already retains the exact main audited LLVM SDK");
    assert.ok(state.inputRows.some(row => row.artifact === "native-linux-rocm-host-sdk-123456-2.tar" && row.roots.includes(paths.sdkName)),
      "the same-run ROCm host input tar must carry that LLVM SDK root");
    assert.deepEqual(nativeSdkConfig(config, root).keys.sort(), ["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"]);
    assert.equal(config.controller.source, path.join(state.hermes.sdkRoot, "source"));
    assert.equal(config.controller.build, path.join(state.hermes.sdkRoot, "build"));
    assert.deepEqual(Object.keys(config.sdkSources).sort(), ["acpp", "cudaImages", "hermes", "openmpSdk", "rocm", "zstd"]);
    assert.equal(config.gpu.dpcpp.env.MOM_AOT_CUDA_IMAGES, config.sdkSources.cudaImages);
    const acppEnv = {CXX: tools.cxx, LLVM_CONFIG: path.join(state.llvmSdk, "bin/llvm-config"),
      MOM_AOT_OPENMP_LIBRARY: path.join(state.openmp.sdkRoot, "sdk/libomp.a"),
      MOM_AOT_ROCM_STATIC_ROOT: path.join(state.rocm.runtimeOutput, "sdk"),
      MOM_AOT_COMGR_RECEIPT: path.join(state.rocm.sdkRoot, "result.json"), MOM_AOT_LLD_ROOT: state.llvmSdk};
    assert.deepEqual(config.gpu["acpp-cuda"].env, {...acppEnv, MOM_AOT_CUDA_IMAGES: config.sdkSources.cudaImages});
    assert.deepEqual(config.gpu["acpp-hip"].env, acppEnv);
    assert.deepEqual(config.sdk.links, ["--ld-path=" + linker, "-Wl,--threads=2", "-Wl,--thinlto-jobs=2",
      "-Wl,-T," + path.join(root, "runtime/static-constructors.ld"), path.join(state.dpcpp.dpcppRoot, "lib/libsycl-devicelib-host.a")]);
    assert.throws(() => linuxNativeSdkDescriptorStatePaths(paths, {...environment,
      MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: "/tmp/stale-linux-descriptor"}), /exactly match/);
  });

  test("selects only the pinned compiler's in-root ld.lld query result", {skip: linuxOnlySkip}, t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-oneapi-linker-")),
      compilerRoot = path.join(base, "compiler", "2026.2"), cxx = path.join(compilerRoot, "bin", "icpx"),
      compilerBin = path.dirname(cxx), linkerDir = path.join(compilerBin, "compiler"), linker = path.join(linkerDir, "ld.lld"),
      target = path.join(linkerDir, "lld");
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    fs.mkdirSync(linkerDir, {recursive: true}); fs.writeFileSync(target, "linker\n", {mode: 0o755}); fs.symlinkSync("lld", linker);
    const query = (/** @type {string} */ stdout, status = 0, stderr = "") =>
      (/** @type {string} */ compiler, /** @type {string[]} */ args) => {
        assert.equal(compiler, cxx); assert.deepEqual(args, ["-print-prog-name=ld.lld"]);
        return {status, signal: null, stdout, stderr};
      };
    assert.equal(linuxNativeSdkLldPath(cxx, query(linker + "\n")), linker, "retain the driver's ld.lld alias instead of realpathing it");
    assert.throws(() => linuxNativeSdkLldPath(cxx, query(linker + "\n", 1)), /query failed/);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query(linker + "\n", 0, "warning\n")), /query failed/);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query("")), /single absolute.*path/);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query("ld.lld\n")), /single absolute.*path/);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query(linker + "\nextra\n")), /single absolute.*path/);
    fs.chmodSync(target, 0o644);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query(linker + "\n")), /not executable/);
    fs.chmodSync(target, 0o755);
    const outside = path.join(base, "outside-lld"); fs.writeFileSync(outside, "outside\n", {mode: 0o755});
    fs.unlinkSync(linker); fs.symlinkSync(outside, linker);
    assert.throws(() => linuxNativeSdkLldPath(cxx, query(linker + "\n")), /within the pinned compiler SDK/);
  });

  test("Linux descriptor input archive rejects missing, redirected, or substituted roots", {skip: linuxOnlySkip}, t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-descriptor-tar-")), incoming = path.join(base, "incoming"),
      source = path.join(base, "source"), rootName = "mom-descriptor-input-123456-2", artifact = "native-input-123456-2.tar",
      archive = path.join(incoming, artifact), rootPath = path.join(source, rootName), row = {artifact, file: archive, roots: [rootName]};
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    fs.mkdirSync(incoming); fs.mkdirSync(rootPath, {recursive: true}); fs.writeFileSync(path.join(rootPath, "receipt.json"), "{}\n");
    packState(archive, source, [rootName]);
    const admitted = auditLinuxNativeSdkDescriptorInputArchive(row, incoming);
    assert.equal(admitted.roots[0], rootName); assert.match(admitted.sha256, /^[a-f0-9]{64}$/);
    assert.throws(() => auditLinuxNativeSdkDescriptorInputArchive({...row, file: path.join(incoming, "redirect", artifact)}, incoming), /path or root/);
    fs.unlinkSync(archive);
    assert.throws(() => auditLinuxNativeSdkDescriptorInputArchive(row, incoming));
    const substituted = path.join(base, "substituted"); fs.mkdirSync(path.join(substituted, "wrong-root"), {recursive: true});
    fs.writeFileSync(path.join(substituted, "wrong-root", "receipt.json"), "substitute\n");
    packState(archive, substituted, ["wrong-root"]);
    assert.throws(() => auditLinuxNativeSdkDescriptorInputArchive(row, incoming), /unexpected state roots/);
  });

  test("component CI validates the profile and seals only a canonical tar holding its manifest", {skip: linuxOnlySkip}, t => {
    const {componentProfile, sealLinuxComponent} = require("../scripts/ci/linux-application");
    for (const profile of linuxNativeComponentProfiles) {assert.equal(componentProfile({NATIVE_PROFILE: profile}), profile);}
    for (const profile of ["", "cpu", "core,oneapi", "../core"]) {
      assert.throws(() => componentProfile({NATIVE_PROFILE: profile}), /exact Linux Core or GPU factory profile/);
    }
    const runnerTemp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-component-seal-")));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const paths = {suffix: "123456-2", runnerTemp}, state = linuxNativeComponentStatePaths(paths, "dpcpp"), base = path.join(runnerTemp, "state"),
      manifest = path.join(base, state.name, "component.json");
    fs.mkdirSync(path.dirname(manifest), {recursive: true});
    fs.writeFileSync(path.join(base, state.name, "object.o"), "object\n");
    packState(state.archive, base, [state.name]);
    assert.throws(() => sealLinuxComponent(paths, "dpcpp"), /missing its producer manifest/);
    fs.unlinkSync(state.archive); fs.writeFileSync(manifest, "{}\n");
    packState(state.archive, base, [state.name]);
    assert.deepEqual(sealLinuxComponent(paths, "dpcpp"), {bundle_sha256: fileHash(state.archive)});
    assert.throws(() => sealLinuxComponent(paths, "oneapi"), /ENOENT/, "each profile seals only its own run-scoped archive");
    const alias = path.join(runnerTemp, "alias.tar"); fs.linkSync(state.archive, alias);
    assert.throws(() => sealLinuxComponent(paths, "dpcpp"), /canonical singly-linked/);
    fs.unlinkSync(alias); fs.unlinkSync(state.archive);
    fs.mkdirSync(path.join(base, "mom-native-linux-component-dpcpp-123456-1"));
    packState(state.archive, base, ["mom-native-linux-component-dpcpp-123456-1"]);
    assert.throws(() => sealLinuxComponent(paths, "dpcpp"), /unexpected state roots/);
  });
});

test.describe("Linux native component handoffs", () => {
  /** @param {import("node:test").TestContext} t */
  function componentFixture(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-component-test-")), suffix = "123456-2",
      profile = "dpcpp", name = "mom-native-linux-component-" + profile + "-" + suffix, rootPath = path.join(base, name),
      gpuRoot = path.join(rootPath, "gpu", profile), receipt = path.join(gpuRoot, "result.json"), object = path.join(gpuRoot, "device-linked.o"),
      admissionFile = path.join(base, "admission.json"), archive = base + ".tar",
      state = {profile, name, root: rootPath, archive, admissionFile};
    t.after(() => {fs.rmSync(base, {recursive: true, force: true}); fs.rmSync(archive, {force: true});});
    fs.mkdirSync(gpuRoot, {recursive: true});
    fs.writeFileSync(object, "device graph\n", {flag: "wx"});
    fs.writeFileSync(receipt, JSON.stringify({factory: "mom_native_gpu_dpcpp", object}) + "\n", {flag: "wx"});
    const admission = {descriptor: {sha256: "a".repeat(64)}};
    fs.writeFileSync(admissionFile, JSON.stringify(admission) + "\n", {flag: "wx"});
    const inputArchives = Array.from({length: 12}, (_, index) => ({artifact: "native-input-" + index + ".tar", sha256: String(index % 10).repeat(64)}));
    const manifest = writeLinuxNativeComponentManifest(state, admission.descriptor.sha256, fileHash(admissionFile), inputArchives,
      {receipt: path.relative(rootPath, receipt), sha256: fileHash(receipt), object, objectSha256: fileHash(object)});
    packState(archive, base, [name]);
    return {state, admission, inputArchives, manifest, receipt, object, archive, members: listState(archive, [name])};
  }

  test("uses only the exact Core-plus-five component set in the derived public config", () => {
    const state = {images: {handoffRoot: "/tmp/native-images"}, acppSdk: {sdkRoot: "/tmp/acpp-sdk"},
        gnu: {name: "mom-native-gnu-headers-test", root: "/tmp/gnu-headers", sdkTar: "/tmp/native-linux-gnu-headers-test.tar"},
        acpp: {preparedCheckout: "/tmp/acpp-source"}, dpcpp: {dpcppRoot: "/tmp/dpcpp", cudaRoot: "/tmp/cuda"},
        jit: {staticComponent: "/tmp/static-dpcpp"}, hermes: {source: "/tmp/hermes-source", sdkRoot: "/tmp/hermes-sdk"},
        rocm: {sdkRoot: "/tmp/rocm-comgr-sdk", runtimeOutput: "/tmp/rocm-host/runtime"},
        openmp: {sdkRoot: "/tmp/openmp-sdk"}, zstd: {sdkRoot: "/tmp/zstd-sdk"},
        opensslSdk: "/tmp/openssl-sdk", ioSdk: "/tmp/io-sdk", llvmSdk: "/tmp/llvm-sdk"},
      linker = "/fixtures/intel-compiler-sdk/ld.lld",
      original = linuxNativeSdkDescriptorConfig(state, {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx",
        cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"}, linker),
      gpu = Object.fromEntries(linuxNativeGpuProfiles.map(profile => [profile, "/tmp/components/" + profile + "/result.json"])),
      components = linuxNativeAssemblyComponents("/tmp/components/core/cpu-core.a", gpu), derived = linuxNativeAssemblyConfig(original, components);
    assert.deepEqual(derived, {...original, components});
    assert.deepEqual(original, linuxNativeSdkDescriptorConfig(state, {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx",
      cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"}, linker), "the admitted descriptor is not mutated");
    assert.doesNotThrow(() => nativeSdkConfig(derived, root));
    const missing = {...gpu}; delete missing["oneapi"];
    assert.throws(() => linuxNativeAssemblyComponents("/tmp/core.a", missing), /exact Core and five/);
    assert.throws(() => linuxNativeAssemblyComponents("/tmp/core.a", {...gpu, unknown: "/tmp/unknown.json"}), /exact Core and five/);
    assert.throws(() => linuxNativeAssemblyConfig({...original, components: {}}, components), /unchanged admitted source SDK descriptor/);
  });

  test("binds each component receipt and regular-file tree to the admitted SDK attempt", {skip: linuxOnlySkip}, t => {
    const f = componentFixture(t), admitted = auditLinuxNativeComponentTree(f.state, f.admission, f.inputArchives, f.members);
    assert.equal(admitted.manifest.profile, "dpcpp");
    assert.equal(admitted.producer.objectSha256, fileHash(f.object));
    assert.equal(admitted.files.length, 2);
    assert.throws(() => auditLinuxNativeComponentTree(f.state, {...f.admission, descriptor: {sha256: "b".repeat(64)}},
      f.inputArchives, f.members), /not bound to this exact SDK descriptor/);
    assert.throws(() => auditLinuxNativeComponentTree(f.state, f.admission,
      f.inputArchives.map((row, index) => index ? row : {...row, sha256: "f".repeat(64)}), f.members), /not bound to this exact SDK descriptor/);
  });

  test("requires exactly the six producer SHA-256 pins", () => {
    const names = linuxNativeComponentProfiles,
      envName = (/** @type {string} */ profile) => "MOM_NATIVE_LINUX_COMPONENT_" + profile.toUpperCase().replaceAll("-", "_") + "_SHA256",
      environment = Object.fromEntries(names.map(profile => [envName(profile), "a".repeat(64)])),
      pins = linuxNativeComponentArchivePins(environment);
    assert.deepEqual(pins, Object.fromEntries(names.map(profile => [profile, "a".repeat(64)])));
    for (const profile of names) {
      const key = envName(profile), missing = {...environment}; delete missing[key];
      assert.throws(() => linuxNativeComponentArchivePins(missing), /all six.*external SHA-256 pins/i, profile);
      assert.throws(() => linuxNativeComponentArchivePins({...environment, [key]: "A".repeat(64)}),
        /all six.*external SHA-256 pins/i, profile);
    }
  });

  test("checks the external component archive pin before parsing or extraction", {skip: linuxOnlySkip}, t => {
    const f = componentFixture(t), admitted = auditLinuxNativeComponentArchive(f.state, f.archive, fileHash(f.archive));
    assert.equal(admitted.archiveSha256, fileHash(f.archive));
    assert.ok(admitted.members.includes(f.state.name + "/component.json"));
    const malformed = f.archive + ".substituted";
    t.after(() => fs.rmSync(malformed, {force: true}));
    fs.writeFileSync(malformed, "not a tar\n", {flag: "wx"});
    assert.throws(() => auditLinuxNativeComponentArchive(f.state, malformed, fileHash(f.archive)), /external SHA-256 pin/i,
      "a mismatched archive is rejected by its producer pin before the tar reader runs");
    assert.throws(() => auditLinuxNativeComponentArchive(f.state, f.archive, "0".repeat(64)), /external SHA-256 pin/i);
    assert.throws(() => auditLinuxNativeComponentArchive(f.state, f.archive, "not-a-digest"), /external SHA-256 pin/i);
  });

  test("rejects missing or extra component members and modified producer objects", {skip: linuxOnlySkip}, t => {
    const f = componentFixture(t), objectMember = f.state.name + "/gpu/dpcpp/device-linked.o";
    assert.throws(() => auditLinuxNativeComponentTree(f.state, f.admission, f.inputArchives,
      f.members.filter(name => name !== objectMember)), /extra, missing or substituted members/);
    assert.throws(() => auditLinuxNativeComponentTree(f.state, f.admission, f.inputArchives, [...f.members, f.state.name + "/extra"]),
      /extra, missing or substituted members/);
    fs.appendFileSync(f.object, "tamper\n");
    assert.throws(() => auditLinuxNativeComponentTree(f.state, f.admission, f.inputArchives, f.members), /files changed after the producer handoff/);
  });

  test("rejects a substituted factory receipt even when paths and tar members look valid", {skip: linuxOnlySkip}, t => {
    const f = componentFixture(t), record = JSON.parse(fs.readFileSync(f.receipt, "utf8"));
    record.factory = "mom_native_gpu_acpp_cuda";
    fs.writeFileSync(f.receipt, JSON.stringify(record) + "\n");
    assert.throws(() => auditLinuxNativeComponentTree(f.state, f.admission, f.inputArchives, f.members));
  });

  test("keeps run-scoped output roots profile-specific and rejects unknown profiles", {skip: linuxOnlySkip}, () => {
    const paths = {suffix: "123456-2", runnerTemp: path.join(os.tmpdir(), "runner-temp")};
    assert.equal(linuxNativeComponentStatePaths(paths, "acpp-cuda").name, "mom-native-linux-component-acpp-cuda-123456-2");
    assert.throws(() => linuxNativeComponentStatePaths(paths, "dpcpp-extra"), /exact GPU factory profile/);
  });

  test("shrinks each child timeout against one aggregate component deadline", () => {
    assert.equal(linuxNativeComponentChildTimeout(10000, 7000, 1000), 7000);
    assert.equal(linuxNativeComponentChildTimeout(10000, 7000, 6000), 4000);
    assert.throws(() => linuxNativeComponentChildTimeout(10000, 7000, 9001), /deadline expired/);
    assert.throws(() => linuxNativeComponentChildTimeout(Number.POSITIVE_INFINITY, 7000, 1000), /deadline is malformed/);
  });

  test("never gives Core or app compiler setup an unbounded zero timeout", () => {
    const deadline = 60 * 60 * 1000;
    for (const reserve of [10 * 60 * 1000, 20 * 60 * 1000]) {
      assert.equal(linuxNativeComponentChildTimeout(deadline, 15 * 60 * 1000, deadline - reserve - 1000 + reserve), 1000);
      assert.throws(() => linuxNativeComponentChildTimeout(deadline, 15 * 60 * 1000, deadline - reserve + 1 + reserve), /deadline expired/);
    }
  });

  test("partitions then restores the descriptor before rejecting a bad SDK input tar", {skip: process.platform !== "linux"}, t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-component-restore-")), run = String(Date.now()), attempt = "1",
      suffix = run + "-" + attempt, tempRoot = /** @type {(name: string) => string} */ (name => "/tmp/" + name + "-" + suffix),
      env = {MOM_NATIVE_CI_SUFFIX: `${run}-${attempt}`, RUNNER_TEMP: runnerTemp, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
        MOM_NATIVE_SOURCE_DIR: tempRoot("mom-native-prerequisites"), MOM_NATIVE_LINUX_SDK_DIR: tempRoot("mom-native-llvm21-sdk"),
        MOM_NATIVE_IO_SOURCE_DIR: tempRoot("mom-native-io-prerequisites"), MOM_NATIVE_IO_SDK_DIR: tempRoot("mom-native-io-openssl-sdk"),
        MOM_NATIVE_LINUX_SOURCES_DIR: tempRoot("mom-native-linux-source-prerequisites"),
        MOM_NATIVE_ZSTD_SOURCE_DIR: tempRoot("mom-native-zstd-prerequisites"), MOM_NATIVE_ZSTD_SDK_DIR: tempRoot("mom-native-zstd-sdk"),
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: tempRoot("mom-native-linux-jit-resources"),
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: tempRoot("mom-native-rocm-comgr-sdk"),
        MOM_NATIVE_ROCM_RESOURCES_DIR: tempRoot("mom-native-rocm-resources"),
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: tempRoot("mom-native-rocm-hip-headers"),
        MOM_NATIVE_ROCM_HOST_SDK_DIR: tempRoot("mom-native-rocm-host-sdk"),
        MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: tempRoot("mom-native-linux-sdk-descriptor"),
        MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: tempRoot("mom-native-dpcpp-cuda-inputs"),
        MOM_NATIVE_NVRTC_LINUX_SOURCE_DIR: tempRoot("mom-native-nvrtc-linux-prerequisites"),
        MOM_NATIVE_NVRTC_LINUX_SDK_DIR: tempRoot("mom-native-nvrtc-linux-sdk"),
        MOM_NATIVE_NVRTC_LINUX_126_SOURCE_DIR: tempRoot("mom-native-nvrtc-linux-12.6-prerequisites"),
        MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR: tempRoot("mom-native-nvrtc-linux-12.6-sdk"),
        MOM_NATIVE_CUDA_IMAGES_DIR: tempRoot("mom-native-cuda-images"), MOM_NATIVE_LINUX_JIT_SDK_DIR: tempRoot("mom-native-linux-jit-sdk"),
        MOM_NATIVE_GNU_HEADERS_DIR: tempRoot("mom-native-gnu-headers"), MOM_NATIVE_ACPP_SOURCE_DIR: tempRoot("mom-native-acpp-prerequisites"),
        MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR: tempRoot("mom-native-acpp-cuda-hip-sdk"),
        MOM_NATIVE_HERMES_SOURCE_DIR: tempRoot("mom-native-hermes-prerequisites"), MOM_NATIVE_HERMES_SDK_DIR: tempRoot("mom-native-hermes-sdk"),
        MOM_NATIVE_OPENMP_SDK_DIR: tempRoot("mom-native-openmp-sdk"), MOM_NATIVE_ROCM_SOURCE_DIR: tempRoot("mom-native-rocm-prerequisites")},
      before = Object.fromEntries(Object.keys(env).map(name => [name, process.env[name]])),
      restoredRoot = tempRoot("mom-native-linux-sdk-descriptor");
    t.after(() => {
      fs.rmSync(restoredRoot, {recursive: true, force: true}); fs.rmSync(runnerTemp, {recursive: true, force: true});
      for (const name of Object.keys(env)) {if (before[name] === undefined) {delete process.env[name];} else {process.env[name] = before[name];}}
    });
    Object.assign(process.env, env);
    const paths = ciPaths(), state = linuxNativeSdkDescriptorStatePaths(paths, paths.env), source = path.join(runnerTemp, "descriptor-source");
    // This few-byte handoff fixture tests restore ordering, not its separately tested disk reserve.
    // A memory-limited container's small /tmp must not hide the intended malformed-tar rejection.
    const fixtureDisk = fs.statfsSync(runnerTemp, {bigint: true});
    t.mock.method(fs, "statfsSync", () => ({...fixtureDisk, bavail: (32n << 30n) / fixtureDisk.bsize}));
    fs.mkdirSync(state.downloads); fs.mkdirSync(path.join(source, state.name), {recursive: true});
    fs.writeFileSync(path.join(source, state.name, "sdk.json"), "{}\n");
    fs.writeFileSync(path.join(source, state.name, "admission.json"), "{}\n");
    packState(path.join(state.downloads, path.basename(state.archive)), source, [state.name]);
    for (const row of state.inputRows) {fs.writeFileSync(path.join(state.downloads, row.artifact), "not a tar\n");}
    const partitioned = partitionLinuxNativeDownloads(paths);
    assert.ok(fs.lstatSync(partitioned.descriptor.archive).isFile(), "partition retains the raw descriptor tar for restore");
    assert.throws(() => restoreLinuxNativeDescriptorHandoff(partitioned.descriptor, paths, []), /Native CI tar list failed/);
    assert.ok(fs.lstatSync(path.join(restoredRoot, "sdk.json")).isFile(), "restore extracted the descriptor before rejecting a bad SDK input tar");
  });
});

test.describe("Linux native application producer handoff", () => {
  /** @param {import("node:test").TestContext} t */
  function lineageFixture(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-application-")), suffix = "123456-2",
      rootPath = path.join(base, "descriptor"), incoming = path.join(base, "incoming");
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    fs.mkdirSync(rootPath); fs.mkdirSync(incoming);
    const gnuSource = path.join(base, "gnu-source"), gnuName = "mom-native-gnu-headers-" + suffix,
      gnuRoot = path.join(gnuSource, gnuName), gnuInclude = path.join(gnuSource, "include"),
      gnuArtifact = "native-linux-gnu-headers-" + suffix + ".tar", gnuInputFile = path.join(incoming, gnuArtifact),
      gnuSdkTar = path.join(base, gnuArtifact);
    fs.mkdirSync(gnuSource);
    /** @type {Array<[string, string[]]>} */ const gnuSpecs = [
      ["libstdc++-13-dev", ["c++/13/vector", "c++/13/backward/hash.h", "x86_64-linux-gnu/c++/13/bits/config.h"]],
      ["libc6-dev", ["x86_64-linux-gnu/sys/types.h"]], ["linux-libc-dev", ["linux/types.h"]],
    ];
    const gnuPackages = gnuSpecs.map(([name, entries]) => {
      const files = entries.map(relative => {
        const file = path.join(gnuInclude, relative); fs.mkdirSync(path.dirname(file), {recursive: true});
        fs.writeFileSync(file, "fixture header " + name + "\n"); return file;
      });
      return {name, version: "1.2.3", architecture: "amd64", files};
    });
    require("../scripts/prepare-native-gnu-headers").prepareLinuxGnuHeaders(gnuInclude, gnuRoot, gnuPackages);
    const gnuNotices = gnuPackages.map(row => {
      const file = path.join(gnuRoot, "licenses", row.name, "copyright"), body = "fixture terms " + row.name + "\n";
      fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, body);
      return {package: row.name, version: row.version, sourcePath: "/usr/share/doc/" + row.name + "/copyright", path: file,
        sha256: createHash("sha256").update(body).digest("hex")};
    });
    fs.writeFileSync(path.join(gnuRoot, "copyrights.json"), JSON.stringify(gnuNotices) + "\n");
    packState(gnuInputFile, gnuSource, [gnuName]);
    const rocmHostName = "mom-native-rocm-host-sdk-" + suffix, rocmSourceRoot = path.join(base, "rocm-source"),
      rocmHostRoot = path.join(rocmSourceRoot, rocmHostName),
      rocmRuntime = path.join(rocmHostRoot, "runtime"), rocmLicenseRelative = "source/rocr/libhsakmt/LICENSE.md",
      rocmSourceLicense = path.join(rocmRuntime, rocmLicenseRelative), rocrArchive = path.join(rocmRuntime, "source-rocr.tar"),
      rocmSdkTar = path.join(base, "native-linux-rocm-host-sdk-" + suffix + ".tar");
    fs.mkdirSync(path.dirname(rocmSourceLicense), {recursive: true});
    fs.writeFileSync(rocmSourceLicense, "fixture hsakmt license\n"); fs.writeFileSync(rocrArchive, "fixture rocr source archive\n");
    const rocmOrigins = {
      drm: {path: "/libdrm/sdk/lib/libdrm.a", sha256: "a".repeat(64), package: "libdrm", architecture: "amd64", version: "2.4.1-1", status: "source"},
      drmAmdgpu: {path: "/libdrm/sdk/lib/libdrm_amdgpu.a", sha256: "b".repeat(64), package: "libdrm", architecture: "amd64", version: "2.4.1-1", status: "source"},
      elf: {path: "/sysroot/root/usr/lib/x86_64-linux-gnu/libelf.a", sha256: "c".repeat(64), package: "libelf-dev", architecture: "amd64", version: "0.1-1", status: "sysroot"},
      numa: {path: "/sysroot/root/usr/lib/x86_64-linux-gnu/libnuma.a", sha256: "d".repeat(64), package: "libnuma-dev", architecture: "amd64", version: "2.0.1-1", status: "sysroot"},
    };
    const stagedBase = "/tmp/mom-native-rocm-sdk-" + suffix + "/sdk/lib/", rocmRecord = {osOrigins: rocmOrigins, stagedDependencies: {
      drm: {...rocmOrigins.drm, staged: stagedBase + "drm.a", stagedSha256: rocmOrigins.drm.sha256},
      drmAmdgpu: {...rocmOrigins.drmAmdgpu, staged: stagedBase + "drmAmdgpu.a", stagedSha256: rocmOrigins.drmAmdgpu.sha256},
      elf: {...rocmOrigins.elf, staged: stagedBase + "elf.a", stagedSha256: rocmOrigins.elf.sha256},
      numa: {...rocmOrigins.numa, staged: stagedBase + "numa.a", stagedSha256: rocmOrigins.numa.sha256}},
    sources: {files: {[rocmLicenseRelative]: fileHash(rocmSourceLicense)}},
    sourceArchives: {rocr: {path: rocrArchive, bytes: fs.statSync(rocrArchive).size, sha256: fileHash(rocrArchive)}}, pins: {rocr: "1".repeat(40)}};
    fs.writeFileSync(path.join(rocmRuntime, "result.json"), JSON.stringify(rocmRecord) + "\n");
    const rocmPackages = rocmHostNoticeRows(rocmRecord).map(row => {
      const file = "licenses/" + row.package + "/copyright", copyrightFile = path.join(rocmHostRoot, file),
        body = "fixture package terms " + row.package + "\n";
      fs.mkdirSync(path.dirname(copyrightFile), {recursive: true}); fs.writeFileSync(copyrightFile, body);
      return {...row, file, sha256: fileHash(copyrightFile), bytes: fs.statSync(copyrightFile).size};
    });
    fs.writeFileSync(path.join(rocmHostRoot, "rocm-notice-sources.json"), JSON.stringify({
      format: "mom-linux-rocm-notice-sources-v1", completeNativeNotices: false,
      runtimeReceipt: {file: "runtime/result.json", sha256: fileHash(path.join(rocmRuntime, "result.json"))}, packages: rocmPackages,
      sourceLicense: {component: "hsakmt-staticdrm", file: "runtime/" + rocmLicenseRelative,
        sha256: rocmRecord.sources.files[rocmLicenseRelative], bytes: fs.statSync(rocmSourceLicense).size,
        sourceArchive: {name: "rocr", sha256: rocmRecord.sourceArchives.rocr.sha256}, revision: rocmRecord.pins.rocr},
    }, null, 2) + "\n");
    packState(rocmSdkTar, rocmSourceRoot, [rocmHostName]);
    /** @type {Array<{artifact:string,file:string,roots:string[],expectedMembers?:string[],recordMembers?:boolean}>} */ const inputRows = [
      {artifact: "native-input-one.tar", file: path.join(incoming, "native-input-one.tar"), roots: ["mom-input-one"]},
      {artifact: "native-input-two.tar", file: path.join(incoming, "native-input-two.tar"), roots: ["mom-input-two"],
        expectedMembers: ["mom-input-two", "mom-input-two/cuda-images.inc", "mom-input-two/cuda-images.inc.json", "mom-input-two/result.json",
          "mom-input-two/notice-sources", "mom-input-two/notice-sources/manifest.json",
          "mom-input-two/notice-sources/cuda-cudart-12.6-copyright.txt", "mom-input-two/notice-sources/cuda-minimal-build-12.6-copyright.txt",
          "mom-input-two/notice-sources/cuda-nvml-dev-12.6-copyright.txt", "mom-input-two/notice-sources/nvrtc-12.6-LICENSE.txt",
          "mom-input-two/notice-sources/nvrtc-13.3-LICENSE.txt", "mom-input-two/notice-sources/nvptxcompiler-13.3-LICENSE.txt",
          "mom-input-two/notice-sources/cutlass-4.6.1-LICENSE.txt"]},
      {artifact: gnuArtifact, file: gnuInputFile, roots: [gnuName], recordMembers: true},
      {artifact: path.basename(rocmSdkTar), file: path.join(incoming, path.basename(rocmSdkTar)), roots: [rocmHostName], recordMembers: true},
    ];
    const firstInput = inputRows[0], secondInput = inputRows[1], rocmInput = inputRows[3];
    assert.ok(firstInput && secondInput && rocmInput);
    fs.writeFileSync(firstInput.file, "same-run-one\n"); fs.writeFileSync(secondInput.file, "same-run-two\n");
    fs.copyFileSync(rocmSdkTar, rocmInput.file);
    const state = {suffix, root: rootPath, name: "mom-native-linux-sdk-descriptor-" + suffix,
      sdkFile: path.join(rootPath, "sdk.json"), admissionFile: path.join(rootPath, "admission.json"), incoming, inputRows,
      runnerTemp: base, downloads: path.join(base, "downloads"), llvmSdk: "/tmp/mom-native-llvm-" + suffix,
      gnu: {name: gnuName, root: gnuRoot, sdkTar: gnuSdkTar},
      images: {handoffRoot: "/tmp/mom-native-images-" + suffix}, acppSdk: {sdkRoot: "/tmp/mom-native-acpp-sdk-" + suffix},
      acpp: {preparedCheckout: "/tmp/mom-native-acpp-source-" + suffix},
      dpcpp: {dpcppRoot: "/tmp/mom-native-dpcpp-" + suffix, cudaRoot: "/tmp/mom-native-cuda-" + suffix},
      jit: {staticComponent: "/tmp/mom-native-jit-static-" + suffix},
      hermes: {source: "/tmp/mom-native-hermes-source-" + suffix, sdkRoot: "/tmp/mom-native-hermes-sdk-" + suffix},
      rocm: {sdkRoot: "/tmp/mom-native-rocm-comgr-sdk-" + suffix, hostSdkName: rocmHostName, hostSdkRoot: rocmHostRoot,
        hostSdkTar: rocmSdkTar, runtimeOutput: rocmRuntime},
      openmp: {sdkRoot: "/tmp/mom-native-openmp-sdk-" + suffix},
      zstd: {sdkRoot: "/tmp/mom-native-zstd-sdk-" + suffix},
      opensslSdk: "/tmp/mom-native-openssl-sdk-" + suffix, ioSdk: "/tmp/mom-native-io-sdk-" + suffix};
    const tools = {cc: "/tools/icx", cxx: "/tools/icpx"}, linker = path.join(base, "intel-compiler-fixture", "ld.lld");
    fs.mkdirSync(path.dirname(linker), {recursive: true});
    fs.writeFileSync(linker, "#!/bin/sh\nexit 0\n", {mode: 0o755}); fs.chmodSync(linker, 0o755);
    const config = linuxNativeSdkDescriptorConfig(state, tools, linker);
    fs.writeFileSync(state.sdkFile, JSON.stringify(config, null, 2) + "\n");
    const inputArchives = inputRows.map(row => ({artifact: row.artifact, bytes: fs.statSync(row.file).size,
        expandedBytes: fs.statSync(row.file).size, sha256: createHash("sha256").update(fs.readFileSync(row.file)).digest("hex"),
        roots: row.roots, ...(row.expectedMembers ? {members: row.expectedMembers.slice().sort()} : {}),
        ...(row.recordMembers ? {members: listState(row.file, row.roots).map(name => name.endsWith("/") ? name.slice(0, -1) : name).sort()} : {})})),
      expandedInputBytes = inputArchives.reduce((sum, row) => sum + row.expandedBytes, 0), reserve = 4 * 1024 * 1024 * 1024;
    const gnuNoticeSources = linuxNativeSdkGnuNoticeSources(state, inputArchives),
      rocmNoticeSources = linuxNativeSdkRocmNoticeSources(state, inputArchives), selectedLinkFiles = {[linker]: fileHash(linker)},
      admission = {format: "mom-native-linux-sdk-descriptor-v1",
        descriptor: {path: state.sdkFile, sha256: createHash("sha256").update(fs.readFileSync(state.sdkFile)).digest("hex")},
        inputArchives, profiles: ["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"],
        diskPreflight: {expandedInputBytes, buildReserveBytes: reserve, requiredTempBytes: expandedInputBytes + reserve,
          availableTempBytes: expandedInputBytes + reserve, availableRunnerTempBytes: reserve},
        gnuNoticeSources, rocmNoticeSources,
        selectedLinkInputs: {files: selectedLinkFiles, sha256: createHash("sha256").update(JSON.stringify(selectedLinkFiles)).digest("hex")},
        buildOnly: true, compiled: false, executed: false, publicationQualified: false};
    fs.writeFileSync(state.admissionFile, JSON.stringify(admission, null, 2) + "\n");
    return {state, tools, linker};
  }

  /** @param {import("node:test").TestContext} t */
  function applicationOutputFixture(t) {
    const {state} = lineageFixture(t), marker = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-app-audit-")),
      suffix = path.basename(marker), name = "mom-native-linux-application-" + suffix, rootPath = path.join(os.tmpdir(), name),
      output = {name, root: rootPath, archive: path.join(state.runnerTemp, name + ".tar"),
        configFile: path.join(state.runnerTemp, "native-linux-assembly-config-" + state.suffix + ".json")};
    t.after(() => {fs.rmSync(rootPath, {recursive: true, force: true}); fs.rmSync(marker, {recursive: true, force: true});});
    fs.mkdirSync(rootPath);
    const binary = path.join(rootPath, "mom"), producerFile = path.join(rootPath, "result.json"), entrypointFile = path.join(rootPath, "entrypoint.json");
    const elf = Buffer.alloc(64);
    elf.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]);
    elf.writeUInt16LE(3, 16); elf.writeUInt16LE(62, 18); elf.writeUInt32LE(1, 20);
    elf.writeBigUInt64LE(64n, 32); elf.writeUInt16LE(64, 52); elf.writeUInt16LE(56, 54); elf.writeUInt16LE(1, 56);
    fs.writeFileSync(binary, elf, {flag: "wx", mode: 0o755}); fs.chmodSync(binary, 0o755);
    const binaryHash = fileHash(binary), bytes = fs.statSync(binary).size,
      producer = {binary, sha256: binaryHash, bytes, configHash: fileHash(state.sdkFile), buildOnly: true,
        publicationQualified: false, application: {application: true, entrypoint: "mom.js", buildOnly: true}};
    fs.writeFileSync(producerFile, JSON.stringify(producer) + "\n", {flag: "wx"});
    const producerHash = fileHash(producerFile), entrypoint = {version: 1, target: "linux", configFile: state.sdkFile,
      configHash: producer.configHash, source: {path: binary, sha256: binaryHash}, artifact: {path: binary, bytes, sha256: binaryHash},
      recipe: {version: 1}, componentHashes: {[producerFile]: producerHash},
      producerReceipt: {path: producerFile, sha256: producerHash}, buildOnly: true, publicationQualified: false};
    fs.writeFileSync(entrypointFile, JSON.stringify(entrypoint) + "\n", {flag: "wx"});
    return {state, output, binary, producerFile, entrypointFile, producer, entrypoint};
  }

  test("measures tar-expanded bytes and requires room for expansion plus build reserve", {skip: process.platform !== "linux"}, t => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-tar-size-")), source = path.join(base, "source"),
      archive = path.join(base, "input.tar"), name = "mom-native-input-123456-2";
    t.after(() => fs.rmSync(base, {recursive: true, force: true}));
    fs.mkdirSync(path.join(source, name), {recursive: true});
    fs.writeFileSync(path.join(source, name, "small.bin"), Buffer.alloc(11));
    fs.writeFileSync(path.join(source, name, "larger.bin"), Buffer.alloc(37));
    packState(archive, source, [name]);
    const members = listState(archive, [name]);
    assert.equal(tarExpandedBytes(archive, members.length), 48);
    const reserve = 4 * 1024 * 1024 * 1024,
      preflight = linuxNativeSdkDiskPreflight(48, 48 + reserve, reserve);
    assert.equal(preflight.requiredTempBytes, 48 + reserve);
    assert.throws(() => linuxNativeSdkDiskPreflight(48, 47 + reserve, reserve), /Insufficient disk/);
    assert.throws(() => linuxNativeSdkDiskPreflight(48, 48 + reserve, reserve - 1), /Insufficient disk/);
  });

  test("binds the exact descriptor, input tar hashes and preserved logical roots", {skip: linuxOnlySkip}, t => {
    const {state, tools, linker} = lineageFixture(t);
    const admitted = auditLinuxNativeSdkDescriptorLineage(state, tools, linker);
    assert.equal(admitted.config.sdkSources.cudaImages, "/tmp/mom-native-images-123456-2/cuda-images.inc");
    assert.equal(admitted.inputArchives.length, 4);
    assert.equal(admitted.gnuNoticeSources.completeNativeNotices, false);
    assert.deepEqual(admitted.gnuNoticeSources.files.map(row => row.package), ["libstdc++-13-dev", "libc6-dev", "linux-libc-dev"]);
    assert.equal(admitted.rocmNoticeSources.completeNativeNotices, false);
    assert.deepEqual(admitted.rocmNoticeSources.packages.map(row => row.package), ["libdrm-dev", "libelf-dev", "libnuma-dev"]);
    assert.deepEqual(admitted.rocmNoticeSources.packages.map(row => row.version), ["2.4.1-1", "0.1-1", "2.0.1-1"]);
    assert.equal(admitted.rocmNoticeSources.sourceLicense.sha256,
      fileHash(path.join(state.rocm.runtimeOutput, "source/rocr/libhsakmt/LICENSE.md")));
    const descriptorAdmission = JSON.parse(fs.readFileSync(state.admissionFile, "utf8")),
      gnuArchive = descriptorAdmission.inputArchives.find((/** @type {{artifact:string,members:string[]}} */ row) => row.artifact === path.basename(state.gnu.sdkTar));
    assert.ok(gnuArchive?.members.includes(state.gnu.name + "/copyrights.json"));
    assert.ok(gnuArchive?.members.includes(state.gnu.name + "/licenses/libc6-dev/copyright"));
    const rocmArchive = descriptorAdmission.inputArchives.find((/** @type {{artifact:string,members:string[]}} */ row) => row.artifact === path.basename(state.rocm.hostSdkTar));
    assert.ok(rocmArchive?.members.includes(state.rocm.hostSdkName + "/rocm-notice-sources.json"));
    assert.ok(rocmArchive?.members.includes(state.rocm.hostSdkName + "/licenses/libnuma-dev/copyright"));
    const gnuBody = path.join(state.gnu.root, admitted.gnuNoticeSources.files[0]?.file || "");
    fs.appendFileSync(gnuBody, "tampered\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(state, tools, linker), /GNU copyright content, package or path changed/);
    const firstInput = state.inputRows[0];
    assert.ok(firstInput);
    fs.appendFileSync(firstInput.file, "substitute\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(state, tools, linker), /input differs/);
  });

  test("rejects selected linker replacement or malformed descriptor link-input admission", {skip: linuxOnlySkip}, t => {
    const replaced = lineageFixture(t);
    fs.writeFileSync(replaced.linker, "#!/bin/sh\nexit 1\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(replaced.state, replaced.tools, replaced.linker),
      /selected linker differs from descriptor admission/);

    const malformed = lineageFixture(t), malformedAdmission = JSON.parse(fs.readFileSync(malformed.state.admissionFile, "utf8"));
    malformedAdmission.selectedLinkInputs.sha256 = "z".repeat(64);
    fs.writeFileSync(malformed.state.admissionFile, JSON.stringify(malformedAdmission) + "\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(malformed.state, malformed.tools, malformed.linker),
      /selected linker differs from descriptor admission/);

    const missing = lineageFixture(t), missingAdmission = JSON.parse(fs.readFileSync(missing.state.admissionFile, "utf8"));
    delete missingAdmission.selectedLinkInputs.files[missing.linker];
    missingAdmission.selectedLinkInputs.sha256 = createHash("sha256").update(JSON.stringify(missingAdmission.selectedLinkInputs.files)).digest("hex");
    fs.writeFileSync(missing.state.admissionFile, JSON.stringify(missingAdmission) + "\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(missing.state, missing.tools, missing.linker),
      /selected linker differs from descriptor admission/);
  });

  test("rejects missing or modified ROCm package and source license bodies", {skip: linuxOnlySkip}, t => {
    const missing = lineageFixture(t), numa = path.join(missing.state.rocm.hostSdkRoot, "licenses/libnuma-dev/copyright");
    fs.unlinkSync(numa);
    assert.throws(() => auditRocmHostNoticeSources(missing.state.rocm), /ROCm package copyright body/);
    const changed = lineageFixture(t), drm = path.join(changed.state.rocm.hostSdkRoot, "licenses/libdrm-dev/copyright");
    fs.appendFileSync(drm, "tampered\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(changed.state, changed.tools, changed.linker), /ROCm package copyright body/);
    const source = lineageFixture(t), license = path.join(source.state.rocm.runtimeOutput, "source/rocr/libhsakmt/LICENSE.md");
    fs.appendFileSync(license, "tampered\n");
    assert.throws(() => auditRocmHostNoticeSources(source.state.rocm), /ROCm hsakmt license body/);
  });

  test("rejects ROCm package versions that no longer match the captured package-origin receipt", {skip: linuxOnlySkip}, t => {
    const {state} = lineageFixture(t), file = path.join(state.rocm.runtimeOutput, "result.json"), record = JSON.parse(fs.readFileSync(file, "utf8"));
    record.osOrigins.numa.version = "2.0.1-2";
    fs.writeFileSync(file, JSON.stringify(record) + "\n");
    assert.throws(() => auditRocmHostNoticeSources(state.rocm), /ROCm static archive origin/);
  });

  test("refuses a missing GNU tar or altered descriptor copyright lineage", {skip: linuxOnlySkip}, t => {
    const missing = lineageFixture(t), gnuRow = missing.state.inputRows.find(row => row.recordMembers);
    assert.ok(gnuRow); fs.unlinkSync(gnuRow.file);
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(missing.state, missing.tools, missing.linker), /missing, stale or substituted/);
    const substituted = lineageFixture(t), substitutedGnu = substituted.state.inputRows.find(row => row.recordMembers);
    assert.ok(substitutedGnu); fs.appendFileSync(substitutedGnu.file, "changed tar bytes\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(substituted.state, substituted.tools, substituted.linker), /input differs/);
    const altered = lineageFixture(t), admission = JSON.parse(fs.readFileSync(altered.state.admissionFile, "utf8"));
    admission.gnuNoticeSources.files[0].sha256 = "0".repeat(64);
    fs.writeFileSync(altered.state.admissionFile, JSON.stringify(admission) + "\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(altered.state, altered.tools, altered.linker), /Same-run package copyright\/source-license lineage/);
  });

  test("refuses missing, extra or stale descriptor inputs", {skip: linuxOnlySkip}, t => {
    const {state, tools, linker} = lineageFixture(t);
    const firstInput = state.inputRows[0];
    assert.ok(firstInput);
    fs.unlinkSync(firstInput.file);
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(state, tools, linker));
    const next = lineageFixture(t);
    fs.writeFileSync(path.join(next.state.incoming, "unexpected.tar"), "extra\n");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(next.state, next.tools, next.linker), /missing, stale or substituted/);
    const changed = lineageFixture(t);
    fs.appendFileSync(changed.state.sdkFile, " ");
    assert.throws(() => auditLinuxNativeSdkDescriptorLineage(changed.state, changed.tools, changed.linker), /descriptor/);
  });

  test("uses one exact public build-only frontdoor and a fresh nonoverlapping output", {skip: linuxOnlySkip}, t => {
    const {state} = lineageFixture(t), paths = linuxNativeApplicationStatePaths({suffix: state.suffix, runnerTemp: state.runnerTemp}),
      nowMs = Date.UTC(2026, 10, 7, 12), deadlineEpoch = String((nowMs + 55 * 60 * 1000) / 1000),
      command = linuxNativeApplicationBuildCommand(state, paths, deadlineEpoch, nowMs);
    assert.equal(paths.name, "mom-native-linux-application-123456-2");
    assert.equal(paths.archive, path.join(state.runnerTemp, "native-linux-application-123456-2.tar"));
    assert.deepEqual(command, {script: "scripts/build-release.js", args: ["--native", state.sdkFile, paths.root], timeoutMs: 50 * 60 * 1000});
    assert.throws(() => linuxNativeApplicationBuildCommand(state, {...paths, root: "/tmp/mom-input-one"}, deadlineEpoch, nowMs), /outside every admitted SDK input/);
  });

  test("application build keeps the merged stage log and copies the linker log head only on failure", {skip: linuxOnlySkip}, async t => {
    const {runApplicationBuild} = require("../scripts/ci/linux-application");
    const runnerTemp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-app-build-"))),
      run = String(Date.now()) + String(process.pid), suffix = run + "-1", linkRoot = path.join("/tmp", "mom-native-linux-application-" + suffix),
      linkLog = path.join(linkRoot, "application", "link.log"), env = {...process.env, MOM_NATIVE_CI_SUFFIX: run + "-1", RUNNER_TEMP: runnerTemp},
      diagnostics = path.join(runnerTemp, "native-linux-application-diagnostics-" + suffix + ".log"),
      linkDiagnostics = path.join(runnerTemp, "native-linux-application-link-diagnostics-" + suffix + ".log"),
      stage = (/** @type {number} */ status) => [process.execPath, "-e",
        `process.stdout.write("stage out\\n"); process.stderr.write("stage err\\n"); process.exitCode = ${status};`];
    t.after(() => {
      fs.rmSync(runnerTemp, {recursive: true, force: true}); fs.rmSync(linkRoot, {recursive: true, force: true});
    });
    fs.mkdirSync(path.dirname(linkLog), {recursive: true});
    fs.writeFileSync(linkLog, Buffer.alloc((16 << 20) + 5, 0x61));
    assert.equal(await runApplicationBuild(env, stage(0)), 0);
    assert.equal(fs.readFileSync(diagnostics, "utf8"), "stage out\nstage err\n", "stderr is merged into the retained stage log");
    assert.equal(fs.existsSync(linkDiagnostics), false, "a successful stage does not copy the linker log");
    assert.equal(await runApplicationBuild(env, stage(3)), 3, "the stage status is preserved");
    assert.equal(fs.statSync(linkDiagnostics).size, 16 << 20, "only the first 16 MiB of the linker log is retained");
    fs.rmSync(linkDiagnostics);
    fs.renameSync(linkLog, linkLog + ".target"); fs.symlinkSync(linkLog + ".target", linkLog);
    assert.equal(await runApplicationBuild(env, stage(1)), 1);
    assert.equal(fs.existsSync(linkDiagnostics), false, "a redirected linker log is not followed");
    await assert.rejects(runApplicationBuild({...env, MOM_NATIVE_CI_SUFFIX: "0/1"}, stage(0)), /MOM_NATIVE_CI_SUFFIX/);
  });

  test("caps public build timeout at the remaining aggregate job deadline", {skip: linuxOnlySkip}, t => {
    const {state} = lineageFixture(t), paths = linuxNativeApplicationStatePaths({suffix: state.suffix, runnerTemp: state.runnerTemp}),
      nowMs = Date.UTC(2026, 10, 7, 12), deadlineEpoch = String((nowMs + 8 * 60 * 1000) / 1000);
    assert.equal(linuxNativeApplicationBuildCommand(state, paths, deadlineEpoch, nowMs).timeoutMs, 8 * 60 * 1000);
  });

  test("rejects expired or malformed aggregate application deadlines", {skip: linuxOnlySkip}, t => {
    const {state} = lineageFixture(t), paths = linuxNativeApplicationStatePaths({suffix: state.suffix, runnerTemp: state.runnerTemp}),
      nowMs = Date.UTC(2026, 10, 7, 12);
    assert.throws(() => linuxNativeApplicationBuildCommand(state, paths, "not-an-epoch", nowMs), /valid job-start deadline/);
    assert.throws(() => linuxNativeApplicationBuildCommand(state, paths, String(Math.floor((nowMs + 5 * 60 * 1000 - 1) / 1000)), nowMs), /Less than five minutes/);
  });

  test("admits the maintained unpacked entrypoint bound to the exact Linux producer and descriptor", {skip: linuxOnlySkip}, t => {
    const f = applicationOutputFixture(t), admitted = auditLinuxNativeApplicationEntrypoint(f.state, f.output, fileHash(f.state.sdkFile));
    assert.equal(admitted.binary, f.binary); assert.equal(admitted.binaryHash, fileHash(f.binary));
    assert.equal(admitted.producerFile, f.producerFile); assert.equal(admitted.entrypointFile, f.entrypointFile);
  });

  test("rejects missing, substituted, or noncanonical application entrypoints", {skip: linuxOnlySkip}, t => {
    const missing = applicationOutputFixture(t);
    fs.unlinkSync(missing.entrypointFile);
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(missing.state, missing.output, fileHash(missing.state.sdkFile)));
    const substituted = applicationOutputFixture(t);
    fs.writeFileSync(substituted.entrypointFile, "{}\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(substituted.state, substituted.output, fileHash(substituted.state.sdkFile)), /entrypoint/);
    const packedSource = applicationOutputFixture(t);
    fs.writeFileSync(packedSource.entrypointFile, JSON.stringify({...packedSource.entrypoint,
      source: {...packedSource.entrypoint.source, path: packedSource.output.root + ".unpacked/mom"}}) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(packedSource.state, packedSource.output,
      fileHash(packedSource.state.sdkFile)));
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(substituted.state,
      {...substituted.output, root: path.join(substituted.output.root, "nested")}, fileHash(substituted.state.sdkFile)), /canonical directory/);
  });

  test("rejects wrong descriptor and artifact bindings in the release entrypoint", {skip: linuxOnlySkip}, t => {
    const f = applicationOutputFixture(t), configHash = fileHash(f.state.sdkFile);
    fs.writeFileSync(f.entrypointFile, JSON.stringify({...f.entrypoint, configHash: "0".repeat(64)}) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(f.state, f.output, configHash), /Native provenance hash differs: descriptor/);
    const aliasConfig = path.join(f.state.root, "same-bytes-different-path.json");
    fs.copyFileSync(f.state.sdkFile, aliasConfig);
    fs.writeFileSync(f.entrypointFile, JSON.stringify({...f.entrypoint, configFile: aliasConfig}) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(f.state, f.output, configHash), /Native application receipt/);
    fs.writeFileSync(f.entrypointFile, JSON.stringify({...f.entrypoint, artifact: {...f.entrypoint.artifact, path: f.producerFile}}) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(f.state, f.output, configHash), /Native entrypoint receipt/);
  });

  test("rejects producer receipt, executable, and component hash drift", {skip: linuxOnlySkip}, t => {
    const f = applicationOutputFixture(t), configHash = fileHash(f.state.sdkFile);
    fs.appendFileSync(f.producerFile, " ");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(f.state, f.output, configHash), /Native provenance hash differs: producer receipt/);
    const changedBinary = applicationOutputFixture(t);
    fs.appendFileSync(changedBinary.binary, "changed\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(changedBinary.state, changedBinary.output,
      fileHash(changedBinary.state.sdkFile)), /entrypoint receipt hash\/size/);
    const wrongComponent = applicationOutputFixture(t);
    fs.writeFileSync(wrongComponent.entrypointFile, JSON.stringify({...wrongComponent.entrypoint,
      componentHashes: {...wrongComponent.entrypoint.componentHashes,
        [wrongComponent.state.sdkFile]: fileHash(wrongComponent.state.sdkFile)}}) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(wrongComponent.state, wrongComponent.output,
      fileHash(wrongComponent.state.sdkFile)), /Native release entrypoint/);
    const wrongRole = applicationOutputFixture(t), producer = {...wrongRole.producer,
      application: {...wrongRole.producer.application, application: false}};
    fs.writeFileSync(wrongRole.producerFile, JSON.stringify(producer) + "\n");
    const producerHash = fileHash(wrongRole.producerFile), entrypoint = {...wrongRole.entrypoint,
      componentHashes: {[wrongRole.producerFile]: producerHash},
      producerReceipt: {path: wrongRole.producerFile, sha256: producerHash}};
    fs.writeFileSync(wrongRole.entrypointFile, JSON.stringify(entrypoint) + "\n");
    assert.throws(() => auditLinuxNativeApplicationEntrypoint(wrongRole.state, wrongRole.output,
      fileHash(wrongRole.state.sdkFile)), /Native application receipt/);
  });
});

test.describe("Windows native Core CI handoff", () => {
  /** @param {import("node:test").TestContext} t */
  function fixture(t) {
    const base = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-native-ci-win-core-")), runnerTemp = path.join(base, "runner-temp");
    t.after(() => fs.rmSync(base, {recursive: true, force: true})); fs.mkdirSync(runnerTemp);
    const environment = {RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: "654321-3"},
      state = windowsNativeCoreStatePaths(environment, "win32"), tools = path.join(base, "tools");
    fs.mkdirSync(state.workRoot); fs.mkdirSync(tools);
    const toolPaths = {MOM_NATIVE_WINDOWS_CORE_NODE_GYP: path.join(tools, "node-gyp.js"),
      MOM_NATIVE_WINDOWS_CORE_MSBUILD: path.join(tools, "MSBuild.exe"), MOM_NATIVE_WINDOWS_CORE_DUMPBIN: path.join(tools, "dumpbin.exe"),
      MOM_NATIVE_WINDOWS_CORE_PYTHON: path.join(tools, "python.exe"), MOM_NATIVE_WINDOWS_CORE_COMPILER: path.join(tools, "icx-cl.exe")};
    for (const file of Object.values(toolPaths)) {fs.writeFileSync(file, "fixture tool\n", {flag: "wx"});}
    fs.mkdirSync(path.join(state.include, "uv"), {recursive: true});
    fs.writeFileSync(path.join(state.include, "uv.h"), "#pragma once\n", {flag: "wx"});
    fs.writeFileSync(path.join(state.include, "uv", "win.h"), "#pragma once\n", {flag: "wx"});
    return {base, state, environment: {...environment, ...toolPaths}};
  }

  test("derives isolated work-root paths and rejects malformed suffixes", t => {
    const f = fixture(t), next = windowsNativeCoreStatePaths({...f.environment, MOM_NATIVE_CI_SUFFIX: "654321-4"}, "win32");
    assert.equal(f.state.suffix, "654321-3");
    assert.equal(path.basename(f.state.archive), "native-windows-core-654321-3.tar");
    assert.equal(path.basename(f.state.bundle), "native-windows-core-654321-3");
    assert.notEqual(f.state.archive, next.archive);
    assert.throws(() => windowsNativeCoreStatePaths(f.environment, "linux"), /Windows runner/);
    assert.throws(() => windowsNativeCoreStatePaths({...f.environment, MOM_NATIVE_CI_SUFFIX: "../654321"}, "win32"), /MOM_NATIVE_CI_SUFFIX/);
  });

  test("requires explicit Intel/MSBuild/Python/Node-gyp tool paths and rejects hostile build flags", t => {
    const f = fixture(t), descriptor = windowsNativeCoreDescriptor(f.state, f.environment);
    assert.equal(descriptor.compiler, f.environment.MOM_NATIVE_WINDOWS_CORE_COMPILER);
    assert.equal(descriptor.libuvInclude, f.state.include);
    assert.throws(() => windowsNativeCoreDescriptor(f.state, {...f.environment, MOM_NATIVE_WINDOWS_CORE_NODE_GYP: "node-gyp.js"}), /nodeGyp/);
    const msvcOnly = path.join(f.base, "cl.exe"); fs.writeFileSync(msvcOnly, "compiler\n");
    assert.throws(() => windowsNativeCoreDescriptor(f.state, {...f.environment, MOM_NATIVE_WINDOWS_CORE_COMPILER: msvcOnly}), /Intel icx-cl/);
    assert.throws(() => windowsNativeCoreDescriptor(f.state, {...f.environment, CL: "/O3"}), /inherited CL/);
    assert.throws(() => windowsNativeCoreDescriptor(f.state, {...f.environment, MOM_NATIVE_WINDOWS_CORE_PYTHON: path.join(f.base, "missing-python.exe")}), /python/);
  });

  test("admits only a source-current portable x64 producer receipt and exact libuv headers", t => {
    const f = fixture(t); fs.mkdirSync(f.state.output);
    const coreFile = path.join(f.state.output, "cpu-core.lib"), receiptFile = path.join(f.state.output, "inputs.json");
    fs.writeFileSync(coreFile, "!<arch>\n", {flag: "wx"});
    const headers = [path.join(f.state.include, "uv.h"), path.join(f.state.include, "uv", "win.h")],
      headerHashes = Object.fromEntries(headers.map(file => [file, fileHash(file)])), sha256 = fileHash(coreFile),
      receipt = {version: 1, target: "win32", gpu: true, portable: true, compiler: {version: "2026.2.0", target: "x86_64-pc-windows-msvc"},
        objectSha256: sha256, objects: [coreFile], archive: {path: coreFile, bytes: fs.statSync(coreFile).size, sha256, coffTableCount: 1, machineCount: 1, x64MachineCount: 1, stubs: false},
        sourceState: nativeSourceInputs(root, "cpu"), headerHashes, buildOnly: true, executed: false, publicationQualified: false};
    fs.writeFileSync(receiptFile, JSON.stringify(receipt) + "\n", {flag: "wx"});
    const admitted = auditWindowsNativeCoreOutput(f.state);
    assert.equal(admitted.archiveSha256, sha256); assert.equal(admitted.receiptSha256, fileHash(receiptFile));
    fs.writeFileSync(receiptFile, JSON.stringify({...receipt, target: "linux"}) + "\n");
    assert.throws(() => auditWindowsNativeCoreOutput(f.state), /portable x64 consumer contract/);
    const windowsHeader = headers[1]; assert.ok(windowsHeader);
    fs.writeFileSync(receiptFile, JSON.stringify(receipt) + "\n"); fs.writeFileSync(windowsHeader, "changed\n");
    assert.throws(() => auditWindowsNativeCoreOutput(f.state), /header input changed/);
  });

  test("caps Windows Core child work to the aggregate job deadline and preserves handoff time", () => {
    const nowMs = Date.UTC(2026, 10, 7, 12);
    /** @param {number} minutes @param {number} [seconds] */
    const deadline = (minutes, seconds = 0) => String((nowMs + minutes * 60 * 1000 + seconds * 1000) / 1000);
    const fetch = windowsNativeCoreDeadlineBudget(deadline(55), "fetch", nowMs),
      after30mInstallAnd5mTests = windowsNativeCoreDeadlineBudget(deadline(20), "fetch", nowMs),
      build = windowsNativeCoreDeadlineBudget(deadline(18), "build", nowMs),
      pack = windowsNativeCoreDeadlineBudget(deadline(7), "package", nowMs);
    assert.equal(fetch.timeoutMs, 20 * 60 * 1000); assert.equal(fetch.remainingMs, 55 * 60 * 1000);
    assert.equal(after30mInstallAnd5mTests.timeoutMs, 4 * 60 * 1000);
    assert.equal(after30mInstallAnd5mTests.remainingMs, 20 * 60 * 1000);
    assert.equal(build.timeoutMs, 11 * 60 * 1000); assert.equal(build.remainingMs, 18 * 60 * 1000);
    assert.equal(pack.timeoutMs, 7 * 60 * 1000); assert.equal(pack.createTimeoutMs, 5 * 60 * 1000);
    assert.equal(pack.listingTimeoutMs, 2 * 60 * 1000); assert.equal(pack.remainingMs, 7 * 60 * 1000);
    assert.throws(() => windowsNativeCoreDeadlineBudget(deadline(16), "fetch", nowMs), /Insufficient aggregate job time remains before fetching/);
    assert.throws(() => windowsNativeCoreDeadlineBudget(deadline(11), "build", nowMs), /Insufficient aggregate job time remains before the Windows Core producer/);
    assert.throws(() => windowsNativeCoreDeadlineBudget(deadline(6, 59), "package", nowMs), /Insufficient aggregate job time remains for the Windows Core handoff/);
  });

  test("rejects malformed, expired, and unbounded Windows Core deadlines", () => {
    const nowMs = Date.UTC(2026, 10, 7, 12), future = String((nowMs + 20 * 60 * 1000) / 1000);
    assert.throws(() => windowsNativeCoreDeadlineBudget("not-an-epoch", "fetch", nowMs), /valid bounded job-start deadline/);
    assert.throws(() => windowsNativeCoreDeadlineBudget(String((nowMs - 1000) / 1000), "fetch", nowMs), /expired or exceeds/);
    assert.throws(() => windowsNativeCoreDeadlineBudget(String((nowMs + 61 * 60 * 1000) / 1000), "build", nowMs), /expired or exceeds/);
    assert.equal(windowsNativeCoreDeadlineBudget(future, "build", nowMs).timeoutMs, 13 * 60 * 1000);
  });

  test("preserves the public producer receipt in the exact three-file tar", {skip: process.platform !== "win32"}, t => {
    const f = fixture(t), coreFile = path.join(f.state.output, "cpu-core.lib"); fs.mkdirSync(f.state.output);
    fs.writeFileSync(coreFile, "!<arch>\n", {flag: "wx"});
    const headers = [path.join(f.state.include, "uv.h"), path.join(f.state.include, "uv", "win.h")],
      hash = fileHash(coreFile), receipt = {version: 1, target: "win32", gpu: true, portable: true,
        compiler: {version: "2026.2.0", target: "x86_64-pc-windows-msvc"}, objectSha256: hash, objects: [coreFile],
        archive: {path: coreFile, bytes: fs.statSync(coreFile).size, sha256: hash, coffTableCount: 1, machineCount: 1, x64MachineCount: 1, stubs: false},
        sourceState: nativeSourceInputs(root, "cpu"), headerHashes: Object.fromEntries(headers.map(file => [file, fileHash(file)])),
        buildOnly: true, executed: false, publicationQualified: false}, receiptFile = path.join(f.state.output, "inputs.json");
    fs.writeFileSync(receiptFile, JSON.stringify(receipt) + "\n", {flag: "wx"});
    const originalReceiptSha = fileHash(receiptFile), report = packWindowsNativeCoreHandoff(f.state, "a".repeat(64), ioPins.libuv.sha256);
    assert.equal(report.core_sha256, hash); assert.equal(report.receipt_sha256, originalReceiptSha);
    assert.deepEqual(fs.readdirSync(f.state.bundle).sort(), ["cpu-core.lib", "handoff.json", "inputs.json"]);
    assert.equal(fileHash(path.join(f.state.bundle, "inputs.json")), originalReceiptSha);
    assert.equal(report.handoff_sha256, fileHash(path.join(f.state.bundle, "handoff.json")));
  });
});

test.describe("Windows host-SDK source/composition CI", () => {
  test("source-only CI import does not require runtime qualification or hash vectors", () => {
    const result = spawnSync(process.execPath, ["--max-old-space-size=128", "-e", `
      const Module = require("node:module"), original = Module._load;
      Module._load = function(request) {
        if (["./check-native-release", "./build-native-windows-core", "./build-native-windows-controller"].includes(request)) {
          throw new Error("Unexpected runtime/mining-build dependency");
        }
        return original.apply(this, arguments);
      };
      require(${JSON.stringify(path.resolve(__dirname, "../scripts/run-native-ci.js"))});
    `], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 0, result.stderr);
  });

  test("derives fresh same-runner paths and rejects mismatched build context", t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-win-host-sdk-"));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const suffix = "987654321-2", workRoot = path.join(fs.realpathSync(runnerTemp), "mom-native-windows-host-sdk-" + suffix),
      outputName = "native-windows-host-sdk-" + suffix,
      environment = {RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: "987654321-2", MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
        MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT: workRoot, MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR: path.join(workRoot, "archives"),
        MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: path.join(workRoot, outputName),
        MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR: path.join(fs.realpathSync(runnerTemp), outputName + "-metadata.tar"),
        MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: "1790000000"},
      state = windowsHostSdkStatePaths(environment, "linux", "x64");
    assert.equal(state.suffix, suffix); assert.equal(state.archiveRoot, environment.MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR);
    assert.equal(state.sdkOutput, environment.MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR); assert.equal(path.basename(state.metadataTar), outputName + "-metadata.tar");
    assert.throws(() => windowsHostSdkStatePaths({...environment, MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: path.join(workRoot, "ambient")}, "linux", "x64"), /paths, job envelope/);
    assert.throws(() => windowsHostSdkStatePaths(environment, "win32", "x64"), /Linux x64 runner/);
    assert.throws(() => windowsHostSdkStatePaths(environment, "linux", "arm64"), /Linux x64 runner/);
    assert.throws(() => windowsHostSdkStatePaths({...environment, MOM_BUILD_JOBS: "3"}, "linux", "x64"), /MOM_BUILD_JOBS set to 2 or 4/);
  });

  test("caps fetch, offline composition, and metadata packaging to the job deadline", () => {
    const nowMs = Date.UTC(2026, 10, 8, 12), deadline = (/** @type {number} */ minutes) => String((nowMs + minutes * 60 * 1000) / 1000),
      fetch = windowsHostSdkDeadlineBudget(deadline(40), "fetch", nowMs), compose = windowsHostSdkDeadlineBudget(deadline(20), "compose", nowMs),
      pack = windowsHostSdkDeadlineBudget(deadline(5), "package", nowMs);
    assert.equal(fetch.timeoutMs, 20 * 60 * 1000); assert.equal(fetch.remainingMs, 40 * 60 * 1000);
    assert.equal(compose.timeoutMs, 16 * 60 * 1000); assert.equal(compose.remainingMs, 20 * 60 * 1000);
    assert.equal(pack.timeoutMs, 4 * 60 * 1000); assert.equal(pack.remainingMs, 5 * 60 * 1000);
    assert.throws(() => windowsHostSdkDeadlineBudget(deadline(12), "compose", nowMs), /Insufficient job time remains/);
    assert.throws(() => windowsHostSdkDeadlineBudget(deadline(61), "fetch", nowMs), /expired or exceeds/);
    assert.throws(() => windowsHostSdkDeadlineBudget("bad", "fetch", nowMs), /bounded job-start deadline/);
  });

  test("metadata artifact names only receipts and never contains vendor SDK payload", () => {
    const lock = windowsHostSdkCatalog();
    assert.equal(Object.keys(lock.packages).length + Object.keys(lock.cabinets).length, 42);
    const state = /** @type {Parameters<typeof windowsHostSdkPackageMembers>[0]} */ ({outputName: "native-windows-host-sdk-123456-1"}),
      members = windowsHostSdkPackageMembers(state);
    assert.deepEqual(members.map(name => name.slice(state.outputName.length + 1)),
      ["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]);
    assert.equal(members.some(name => /(?:^|\/)sdk\//.test(name)), false);
  });
});

test.describe("Windows LLVM descriptor composition", () => {
  test("keeps caller library policy ordered and rejects malformed or duplicated basenames", () => {
    const policy = {msvcLibraries: ["libcmt.lib", "libircmt.lib"], ucrtLibraries: ["libucrt.lib"], importLibraries: ["kernel32.lib"]};
    assert.deepEqual(validateWindowsLlvmLibraryPolicy(policy), policy);
    assert.throws(() => validateWindowsLlvmLibraryPolicy({...policy, importLibraries: ["kernel32.lib", "KERNEL32.LIB"]}), /duplicate/i);
    assert.throws(() => validateWindowsLlvmLibraryPolicy({...policy, msvcLibraries: ["../libcmt.lib"]}), /Unsafe|unsafe/);
    const invalid = {windowsSdkRoot: "/not-opened", windowsSdkPins: {resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64),
      filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)}, hermesRoot: "/not-opened",
    hermesExpectation: {kind: /** @type {"hermes"} */ ("hermes"), handoffSha256: "e".repeat(64)},
    tools: {clangCl: {path: "/not-opened/clang-cl", sha256: "a".repeat(64)}, lldLink: {path: "/not-opened/lld-link", sha256: "b".repeat(64)},
      llvmNm: {path: "/not-opened/llvm-nm", sha256: "c".repeat(64)}, clangResourceInclude: "/not-opened/resource"},
    libraryPolicy: {...policy, msvcLibraries: ["bad?.lib"]}};
    assert.throws(() => composeWindowsLlvmSdkDescriptor(invalid), /Unsafe|unsafe/,
      "library policy is rejected before opening caller-supplied input roots");
    assert.throws(() => composeLocalWindowsLlvmSdkDescriptor({...invalid, hermesExpectation: {kind: "hermes",
      localBuildId: "12345678-1234-4234-9234-123456789abc", handoffSha256: "e".repeat(64)}}), /Unsafe|unsafe/);
  });

  test("local and CI descriptor entrypoints cannot substitute transport identities", () => {
    const input = {windowsSdkRoot: "/not-opened", windowsSdkPins: {resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64),
        filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)}, hermesRoot: "/not-opened",
      tools: {clangCl: {path: "/not-opened/icx", sha256: "a".repeat(64)}, lldLink: {path: "/not-opened/lld-link", sha256: "b".repeat(64)},
        llvmNm: {path: "/not-opened/llvm-nm", sha256: "c".repeat(64)}, clangResourceInclude: "/not-opened/resource"}},
      ci = {kind: "hermes", handoffSha256: "e".repeat(64)},
      local = {kind: "hermes", localBuildId: "12345678-1234-4234-9234-123456789abc", handoffSha256: "e".repeat(64)};
    assert.throws(() => composeWindowsLlvmSdkDescriptor(/** @type {any} */ ({...input, hermesExpectation: local})), /explicit source pins/);
    assert.throws(() => composeLocalWindowsLlvmSdkDescriptor(/** @type {any} */ ({...input, hermesExpectation: ci})), /explicit source pins/);
  });

  test("hashes the explicit LLVM tool files and only accepts a declared sibling alias", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-win-llvm-tools-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const bytes = Buffer.from("pinned LLVM tool\n"), sha256 = createHash("sha256").update(bytes).digest("hex"), tool = path.join(root, "llvm-nm");
    fs.writeFileSync(tool, bytes, {flag: "wx"});
    assert.deepEqual(auditWindowsLlvmToolFile({path: tool, sha256}, "llvmNm"), {path: tool, resolvedPath: tool, bytes: bytes.length, sha256});
    assert.throws(() => auditWindowsLlvmToolFile({path: tool, sha256: "0".repeat(64)}, "llvmNm"), /bytes or canonical path/);
    if (process.platform !== "win32") {
      const alias = path.join(root, "lld-link"); fs.symlinkSync("llvm-nm", alias);
      assert.equal(auditWindowsLlvmToolFile({path: alias, sha256, linkTarget: "llvm-nm"}, "lldLink").resolvedPath, tool);
      assert.throws(() => auditWindowsLlvmToolFile({path: alias, sha256, linkTarget: "../llvm-nm"}, "lldLink"), /alias differs/);
    }
  });

  test("maps selected SDK libraries by case-insensitive path without losing caller order", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-win-llvm-libs-")), sdk = path.join(root, "sdk");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const files = new Map(), make = (/** @type {string} */ relative, /** @type {string} */ text, /** @type {string} */ name) => {
      const full = path.join(sdk, ...relative.split("/")), bytes = Buffer.from("!<arch>\n" + text), sha256 = createHash("sha256").update(bytes).digest("hex");
      fs.mkdirSync(path.dirname(full), {recursive: true}); fs.writeFileSync(full, bytes, {flag: "wx"});
      const row = {path: relative, bytes: bytes.length, sha256, origins: [{package: "sdk-test", packageSha256: "a".repeat(64),
        member: "Package/" + name, sha256, bytes: bytes.length}]};
      files.set(relative.toLowerCase(), row); return {full, row};
    };
    fs.mkdirSync(sdk, {recursive: true});
    const first = make("sdk/lib/um/KERNEL32.lib", "one", "kernel32.lib"), second = make("sdk/lib/um/advapi32.lib", "two", "advapi32.lib");
    const rows = mapWindowsLlvmSdkLibraries(["kernel32.lib", "advapi32.lib"], ["sdk/lib/um"], files, sdk, "import");
    assert.deepEqual(rows.map(row => row.path), [first.full, second.full]);
    assert.deepEqual(rows.map(row => row.manifestPath), [first.row.path, second.row.path]);
    assert.deepEqual(rows.map(row => row.sha256), [first.row.sha256, second.row.sha256]);
    const duplicate = make("msvc/lib/kernel32.lib", "three", "kernel32.lib");
    assert.throws(() => mapWindowsLlvmSdkLibraries(["kernel32.lib"], ["sdk/lib/um", "msvc/lib"], files, sdk, "UCRT"), /exactly one/);
    fs.writeFileSync(duplicate.full, "changed");
    assert.throws(() => mapWindowsLlvmSdkLibraries(["kernel32.lib"], ["msvc/lib"], files, sdk, "MSVC"), /changed or is not regular/);
  });
});

test.describe("Windows LLVM descriptor CI producer", () => {
  const skipLinuxFixedRootTests = {skip: linuxFixedTmpSkip};
  /** @param {import('node:test').TestContext} t */
  const fixture = t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-win-llvm-descriptor-")), run = String(Date.now()) + String(process.pid),
      suffix = run + "-1", outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary"),
      windowsSdkRoot = path.join(runnerTemp, "mom-native-windows-host-sdk-" + suffix, "native-windows-host-sdk-" + suffix),
      hermesRoot = path.join(runnerTemp, "native-windows-controller-hermes-" + suffix), toolRoot = path.join(runnerTemp, "tools"),
      deadlineEpoch = String(Math.ceil((Date.now() + 55 * 60 * 1000) / 1000));
    fs.mkdirSync(windowsSdkRoot, {recursive: true}); fs.mkdirSync(hermesRoot, {recursive: true}); fs.mkdirSync(path.join(toolRoot, "resource"), {recursive: true});
    fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
    /** @param {string} name @param {string} contents */
    const toolFile = (name, contents) => {
      const file = path.join(toolRoot, name), bytes = Buffer.from(contents); fs.writeFileSync(file, bytes, {flag: "wx"});
      return {path: file, sha256: createHash("sha256").update(bytes).digest("hex")};
    };
    const input = {windowsSdkRoot, windowsSdkPins: {resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64),
      filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)}, hermesRoot,
    hermesExpectation: {kind: /** @type {"hermes"} */ ("hermes"), handoffSha256: "e".repeat(64)},
    tools: {clangCl: toolFile("clang-cl", "clang-cl fixture\n"), lldLink: toolFile("lld-link", "lld-link fixture\n"),
      llvmNm: toolFile("llvm-nm", "llvm-nm fixture\n"), clangResourceInclude: path.join(toolRoot, "resource")}};
    const inputFile = path.join(runnerTemp, "descriptor-input.json"); fs.writeFileSync(inputFile, JSON.stringify(input) + "\n", {flag: "wx"});
    const environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: run + "-1",
      MOM_NATIVE_WINDOWS_LLVM_DIR: path.join(os.tmpdir(), "mom-native-windows-llvm21-" + suffix),
      MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(runnerTemp, "mom-native-windows-llvm21-descriptor-" + suffix),
      MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: deadlineEpoch, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
      MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 7 * 60 * 60 * 1000) / 1000)), GITHUB_JOB: "llvm_configure",
      GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile});
    const state = windowsNativeLlvmDescriptorCiStatePaths(environment, "linux", "x64"), sha = (/** @type {string} */ file) => fileHash(file);
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    return {runnerTemp, suffix, input, inputFile, windowsSdkRoot, hermesRoot, outputFile, summaryFile, environment, state, sha};
  };

  test("composes once per identity check and publishes only build-only descriptor provenance", skipLinuxFixedRootTests, t => {
    const f = fixture(t), calls = [], composed = {descriptor: {version: 1, fixture: true}, provenance: {role: "validated-composer-fixture"}},
      composer = (/** @type {unknown} */ input) => {calls.push(input); return composed;}, result = buildWindowsNativeLlvmDescriptorCi([f.inputFile, f.sha(f.inputFile)],
        f.environment, "linux", "x64", Date.now(), composer);
    assert.equal(calls.length, 2, "the maintained composer is reused for the final source/tool identity check");
    assert.equal(result.windows_llvm_descriptor_path, path.join(f.state.root, "descriptor.json"));
    assert.equal(path.dirname(result.windows_llvm_descriptor_path), f.state.root);
    assert.notEqual(f.state.root, f.state.configureRoot, "descriptor output stays outside the configure state root");
    assert.equal(f.sha(result.windows_llvm_descriptor_path), result.windows_llvm_descriptor_sha256);
    assert.equal(f.sha(result.windows_llvm_descriptor_provenance_path), result.windows_llvm_descriptor_provenance_sha256);
    const provenance = JSON.parse(fs.readFileSync(result.windows_llvm_descriptor_provenance_path, "utf8"));
    assert.equal(provenance.input.sha256, f.sha(f.inputFile));
    assert.equal(provenance.buildOnly, true); assert.equal(provenance.compiled, false); assert.equal(provenance.executed, false);
    assert.equal(provenance.publicationQualified, false); assert.equal(fs.existsSync(path.join(f.state.root, "output")), false);
    assert.match(fs.readFileSync(f.outputFile, "utf8"), /windows_llvm_descriptor_provenance_sha256=/);
    assert.match(fs.readFileSync(f.summaryFile, "utf8"), /compiled: false; executed: false; publication-qualified: false/);
  });

  test("rejects stale pins, malformed schema, wrong run roots and missing Hermes runtime before output", skipLinuxFixedRootTests, t => {
    const f = fixture(t), composer = () => {throw new Error("composer must not run for invalid inputs");};
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight([f.inputFile, "0".repeat(64)], f.environment, "linux", "x64"), /external SHA-256 pin/);
    const writeInput = (/** @type {unknown} */ value) => fs.writeFileSync(f.inputFile, JSON.stringify(value) + "\n");
    writeInput({...f.input, hermesExpectation: {...f.input.hermesExpectation, runAttempt: "999-9"}});
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64", Date.now(), composer), /exact same-run roots and handoff/);
    writeInput({...f.input, windowsSdkRoot: path.join(f.runnerTemp, "other-sdk")});
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64"), /exact same-run roots and handoff/);
    writeInput({...f.input, hermesRoot: path.join(f.runnerTemp, "other-hermes")});
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64"), /exact same-run roots and handoff/);
    writeInput({...f.input, extra: true});
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64"), /unexpected composer schema/);
    writeInput(f.input);
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64", Date.now()), /ENOENT|handoff/);
    assert.equal(fs.existsSync(f.state.root), false, "missing same-run Hermes handoff never creates output");
  });

  test("rejects report/output overlap, existing outputs, and an exhausted shared deadline before mutation", skipLinuxFixedRootTests, t => {
    const f = fixture(t), composer = () => {throw new Error("composer must not run for invalid preflight");}, args = [f.inputFile, f.sha(f.inputFile)];
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi(args, {...f.environment, GITHUB_OUTPUT: f.inputFile}, "linux", "x64", Date.now(), composer),
      /reports must not overlap pinned inputs/);
    fs.mkdirSync(f.state.root, {recursive: true});
    const nestedSummary = path.join(f.state.root, "summary"); fs.writeFileSync(nestedSummary, "", {flag: "wx"});
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi(args, {...f.environment, GITHUB_STEP_SUMMARY: nestedSummary}, "linux", "x64", Date.now(), composer),
      /reports must not overlap pinned inputs/);
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi(args, f.environment, "linux", "x64", Date.now(), composer), /fresh and separate/);
    fs.rmSync(f.state.root, {recursive: true});
    const nearDeadline = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 4 * 60 * 1000) / 1000))};
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight(args, nearDeadline, "linux", "x64"), /Insufficient shared job time/);
    assert.equal(fs.existsSync(f.state.root), false, "failed preflights leave descriptor and configure roots untouched");
  });

  test("descriptor composition respects an opted-in aggregate run deadline before output", skipLinuxFixedRootTests, t => {
    const f = fixture(t), now = Date.now(), environment = {...f.environment, GITHUB_JOB: "llvm_descriptor",
      MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(Math.floor((now + 30 * 60 * 1000) / 1000))};
    assert.throws(() => windowsNativeLlvmDescriptorCiPreflight([f.inputFile, f.sha(f.inputFile)], environment, "linux", "x64", now),
      /exceeds the original run deadline/);
    assert.equal(fs.existsSync(f.state.root), false);
    assert.equal(fs.existsSync(f.state.configureRoot), false);
  });

  test("detects a changed compiler after composition without retaining an output", skipLinuxFixedRootTests, t => {
    const f = fixture(t), calls = [], composer = () => {
      calls.push(true); if (calls.length === 1) {fs.writeFileSync(f.input.tools.clangCl.path, "changed compiler\n");}
      return {descriptor: {version: 1}, provenance: {role: "fixture"}};
    };
    assert.throws(() => buildWindowsNativeLlvmDescriptorCi([f.inputFile, f.sha(f.inputFile)], f.environment, "linux", "x64", Date.now(), composer),
      /tool bytes or canonical path differ from their pin/);
    assert.equal(fs.existsSync(f.state.root), false, "tool drift is caught before creating output");
  });

  test("dispatches only the explicit descriptor producer form", async () => {
    await assert.rejects(runStage(["windows-native-llvm21-descriptor"]), /input JSON and its SHA-256 pin/);
    await assert.rejects(runStage(["windows-native-llvm21-descriptor", "/input.json"]), /input JSON and its SHA-256 pin/);
  });
});

test.describe("Windows LLVM 21 CI phase frontdoor", () => {
  const skipLinuxFixedRootTests = {skip: linuxFixedTmpSkip};
  /** @param {import('node:test').TestContext} t */
  const fixture = t => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-win-llvm-ci-")),
      run = String(Date.now()) + String(process.pid), suffix = run + "-1", inputRoot = path.join(runnerTemp, "inputs"), toolRoot = path.join(inputRoot, "tools");
    fs.mkdirSync(toolRoot, {recursive: true});
    const outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary");
    fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
    const sdkRoot = path.join(inputRoot, "windows-sdk"), includeRoot = path.join(sdkRoot, "include"),
      overlay = path.join(inputRoot, "overlay.json"), descriptorFile = path.join(inputRoot, "windows-llvm.json"),
      cmake = path.join(toolRoot, "cmake"), ninja = path.join(toolRoot, "ninja"),
      archive = path.join(inputRoot, windowsLlvmArchiveName), root = path.join(os.tmpdir(), "mom-native-windows-llvm21-" + suffix),
      deadlineEpoch = String(Math.ceil((Date.now() + 55 * 60 * 1000) / 1000));
    fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, roots: [
      {type: "file", name: path.join(includeRoot, "windows.h"), "external-contents": path.join(inputRoot, "windows.h")},
    ]}), {flag: "wx"});
    const descriptor = {version: 1, tools: {clangCl: path.join(toolRoot, "clang-cl"), lldLink: path.join(toolRoot, "lld-link"),
      llvmNm: path.join(toolRoot, "llvm-nm"), clangResourceInclude: path.join(toolRoot, "resource")},
    windowsSdk: {overlay, sourceRoot: path.join(sdkRoot, "source"), includeRoots: [includeRoot], libraryRoots: [path.join(sdkRoot, "lib")],
      msvcLibraries: [path.join(sdkRoot, "lib", "libcmt.lib")], ucrtLibraries: [path.join(sdkRoot, "lib", "libucrt.lib")],
      importLibraries: [path.join(sdkRoot, "lib", "kernel32.lib")]}};
    fs.writeFileSync(descriptorFile, JSON.stringify(descriptor) + "\n", {flag: "wx"});
    const toolFixtures = /** @type {Array<[string,string]>} */ ([[cmake, "cmake tool fixture\n"], [ninja, "ninja tool fixture\n"],
      [archive, "not the pinned archive\n"]]);
    for (const [file, bytes] of toolFixtures) {
      fs.writeFileSync(file, bytes, {flag: "wx"});
    }
    const environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: run + "-1",
        MOM_NATIVE_WINDOWS_LLVM_DIR: root, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: deadlineEpoch, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
        MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 7 * 60 * 60 * 1000) / 1000)), GITHUB_JOB: "llvm_configure",
        GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile}),
      state = windowsNativeLlvmCiStatePaths(environment, "linux", "x64");
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const sha = /** @param {string} file */ file => fileHash(file), configureArgs = ["configure", descriptorFile, sha(descriptorFile), archive, windowsLlvmArchiveSha256,
      cmake, sha(cmake), ninja, sha(ninja)];
    return {runnerTemp, inputRoot, descriptorFile, archive, cmake, ninja, outputFile, summaryFile, environment, state, configureArgs, sha};
  };

  test("uses a fixed same-run output root and rejects redirects before mutation", skipLinuxFixedRootTests, t => {
    const f = fixture(t);
    assert.equal(f.state.root, f.environment["MOM_NATIVE_WINDOWS_LLVM_DIR"]);
    assert.equal(f.state.output, path.join(f.state.root, "output"));
    assert.equal(f.state.deadlineRecord, path.join(f.state.root, "ci-job-deadline-llvm_configure.json"));
    // Every phase needs the original run deadline; there is no single-job record that a later phase would recreate.
    const singleJob = {...f.environment}; delete singleJob["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"];
    assert.throws(() => windowsNativeLlvmCiStatePaths(singleJob, "linux", "x64"), /original run deadline/);
    assert.throws(() => windowsNativeLlvmCiStatePaths({...f.environment, MOM_NATIVE_WINDOWS_LLVM_DIR: path.join(f.runnerTemp, "ambient")}, "linux", "x64"), /CI work-root plan/);
    assert.throws(() => windowsNativeLlvmCiStatePaths(f.environment, "win32", "x64"), /Linux x64 runner/);
    assert.throws(() => windowsNativeLlvmCiStatePaths({...f.environment, MOM_BUILD_JOBS: "3"}, "linux", "x64"), /MOM_BUILD_JOBS set to 2 or 4/);
    const wrongDescriptorPin = [...f.configureArgs]; wrongDescriptorPin[2] = "0".repeat(64);
    assert.throws(() => buildWindowsNativeLlvmCi(wrongDescriptorPin, f.environment),
      /descriptor differs from its external SHA-256 pin/);
    assert.throws(() => buildWindowsNativeLlvmCi([...f.configureArgs.slice(0, 4), "0".repeat(64), ...f.configureArgs.slice(5)], f.environment),
      /exact locked LLVM 21 source archive/);
    assert.equal(fs.existsSync(f.state.root), false, "bad external pins never create the output root");
    if (process.platform !== "win32") {
      const redirected = path.join(f.inputRoot, "redirected.json"); fs.symlinkSync(f.descriptorFile, redirected);
      const args = [...f.configureArgs]; args[1] = redirected; args[2] = f.sha(f.descriptorFile);
      assert.throws(() => buildWindowsNativeLlvmCi(args, f.environment), /canonical same-run input/);
      assert.equal(fs.existsSync(f.state.root), false);
    }
  });

  test("rejects workflow reports that would overwrite an input", skipLinuxFixedRootTests, t => {
    const f = fixture(t), descriptorHash = f.sha(f.descriptorFile), configureEnvironment = {...f.environment,
      GITHUB_OUTPUT: f.descriptorFile};
    assert.throws(() => buildWindowsNativeLlvmCi(f.configureArgs, configureEnvironment), /reports must not overlap pinned inputs/);
    assert.equal(f.sha(f.descriptorFile), descriptorHash);
    assert.equal(fs.existsSync(f.state.root), false, "overlapping report paths are rejected before output creation");

    fs.mkdirSync(f.state.root, {mode: 0o700}); windowsNativeLlvmCiDeadlineRecord(f.state, true); fs.mkdirSync(f.state.output);
    fs.copyFileSync(f.descriptorFile, f.state.descriptor);
    const stagedHash = f.sha(f.state.descriptor), receipt = {role: "windows-llvm21-host-sdk", stage: "configured",
      buildOnly: true, compiled: false, executed: false, publicationQualified: false,
      plan: {output: f.state.output, descriptorFile: f.state.descriptor, archive: f.archive, tools: {cmake: f.cmake, ninja: f.ninja}, jobs: 2},
      fingerprints: {[f.state.descriptor]: stagedHash}};
    fs.writeFileSync(f.state.result, JSON.stringify(receipt) + "\n", {flag: "wx", mode: 0o444});
    const buildEnvironment = {...f.environment, GITHUB_OUTPUT: f.archive};
    assert.throws(() => windowsNativeLlvmCiPreflight(["build", "llvm-amdgpu"], buildEnvironment), /reports must not overlap pinned inputs/);
    assert.equal(f.sha(f.state.descriptor), stagedHash);
  });

  test("bounds configure, build and audit using one absolute deadline", () => {
    const now = Date.UTC(2026, 10, 8, 12), epoch = (/** @type {number} */ minutes) => String((now + minutes * 60 * 1000) / 1000),
      first = windowsNativeLlvmCiDeadlineBudget(epoch(55), "build", now);
    assert.equal(first.timeoutMs, 45 * 60 * 1000);
    assert.equal(windowsNativeLlvmCiDeadlineBudget(epoch(55), "configure", now).timeoutMs, first.timeoutMs);
    assert.throws(() => windowsNativeLlvmCiDeadlineBudget(epoch(55), "build", now + 50 * 60 * 1000), /Insufficient shared job time/);
    assert.throws(() => windowsNativeLlvmCiDeadlineBudget(epoch(61), "audit", now), /60-minute job bound/);
    assert.throws(() => windowsNativeLlvmCiDeadlineBudget("expired", "configure", now), /valid shared job-start deadline/);
  });

  test("refuses a reset deadline between successive phase calls", skipLinuxFixedRootTests, t => {
    const f = fixture(t), state = f.state, original = f.environment["MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH"] || "";
    fs.mkdirSync(state.root, {mode: 0o700}); windowsNativeLlvmCiDeadlineRecord(state, true); fs.mkdirSync(state.output);
    const changed = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Number(original) + 60)};
    assert.throws(() => windowsNativeLlvmCiPreflight(["build", "llvm-amdgpu"], changed, "linux", "x64"), /replace or reset its original job deadline/);
    assert.equal(fs.existsSync(state.result), false);
  });

  test("cross-job limits fail before configure mutation and keep the run inside eight hours", skipLinuxFixedRootTests, t => {
    const f = fixture(t), now = Date.now(), epoch = (/** @type {number} */ minutes) => String(Math.ceil((now + minutes * 60 * 1000) / 1000)),
      valid = {...f.environment, GITHUB_JOB: "llvm_configure", MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: epoch(7 * 60),
        MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(55)};
    assert.deepEqual(windowsNativeLlvmCiRunDeadlineBudget(epoch(7 * 60), now), {remainingMs: Number(epoch(7 * 60)) * 1000 - now});
    assert.equal(windowsNativeLlvmCiDeadlineBudget(epoch(55), "build", now, epoch(7 * 60)).timeoutMs, 45 * 60 * 1000);
    assert.throws(() => windowsNativeLlvmCiRunDeadlineBudget(epoch(-1), now), /run deadline is expired/);
    assert.throws(() => windowsNativeLlvmCiRunDeadlineBudget(epoch(8 * 60 + 1), now), /8-hour run bound/);

    const invalidCases = /** @type {Array<[NodeJS.ProcessEnv,RegExp]>} */ ([
      [{...valid, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: epoch(-1)}, /run deadline is expired/],
      [{...valid, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: epoch(8 * 60 + 1)}, /8-hour run bound/],
      [{...valid, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(-1)}, /deadline is expired/],
      [{...valid, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(61)}, /60-minute job bound/],
      [{...valid, MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: epoch(30), MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(45)}, /exceeds the original run deadline/],
      [{...valid, GITHUB_JOB: "bad/job"}, /safe bounded GITHUB_JOB identity/],
      [{...valid, GITHUB_JOB: "x".repeat(65)}, /safe bounded GITHUB_JOB identity/],
    ]);
    for (const [environment, pattern] of invalidCases) {
      assert.throws(() => windowsNativeLlvmCiPreflight(f.configureArgs, environment, "linux", "x64", now), pattern);
      assert.equal(fs.existsSync(f.state.root), false, "invalid cross-job deadlines leave the LLVM state root untouched");
    }
    const split = windowsNativeLlvmCiStatePaths(valid, "linux", "x64");
    assert.equal(split.deadlineRecord, path.join(split.root, "ci-job-deadline-llvm_configure.json"));
    assert.equal(split.runDeadlineRecord, path.join(split.root, "ci-run-deadline.json"));
  });

  test("pins first job deadline, rejects record tampering, and permits a later job only on the same run limit", skipLinuxFixedRootTests, t => {
    const f = fixture(t), now = Date.now(), epoch = (/** @type {number} */ minutes) => String(Math.ceil((now + minutes * 60 * 1000) / 1000)),
      runDeadlineEpoch = epoch(7 * 60), firstEnvironment = {...f.environment, GITHUB_JOB: "llvm_configure",
        MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: runDeadlineEpoch},
      firstState = windowsNativeLlvmCiStatePaths(firstEnvironment, "linux", "x64");
    fs.mkdirSync(firstState.root, {mode: 0o700});
    const runDeadlineRecord = firstState.runDeadlineRecord;
    if (!runDeadlineRecord) {throw new Error("Cross-job fixture lacks its global deadline record path");}
    const first = windowsNativeLlvmCiDeadlineRecord(firstState, true), runBytes = fs.readFileSync(runDeadlineRecord, "utf8");
    assert.deepEqual(first, {version: 1, job: "llvm_configure",
      deadlineEpoch: firstState.deadlineEpoch, runDeadlineEpoch});
    assert.equal(JSON.parse(runBytes).runDeadlineEpoch, runDeadlineEpoch);
    assert.throws(() => windowsNativeLlvmCiDeadlineRecord({...firstState, deadlineEpoch: epoch(56)}), /replace or reset its original job deadline/);
    assert.throws(() => windowsNativeLlvmCiDeadlineRecord({...firstState, runDeadlineEpoch: epoch(7 * 60 + 1)}), /replace or reset its original run deadline/);

    const secondEnvironment = {...firstEnvironment, GITHUB_JOB: "llvm_build_lld", MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: epoch(45)},
      secondState = windowsNativeLlvmCiStatePaths(secondEnvironment, "linux", "x64");
    const secondRunDeadlineRecord = secondState.runDeadlineRecord;
    if (!secondRunDeadlineRecord) {throw new Error("Later-job fixture lacks its global deadline record path");}
    const second = windowsNativeLlvmCiDeadlineRecord(secondState, true);
    assert.equal(second.job, "llvm_build_lld"); assert.notEqual(second.deadlineEpoch, first.deadlineEpoch);
    assert.equal(fs.readFileSync(secondRunDeadlineRecord, "utf8"), runBytes, "later jobs preserve the original global limit byte-for-byte");
    assert.throws(() => windowsNativeLlvmCiDeadlineRecord({...secondState, deadlineEpoch: epoch(46)}), /replace or reset its original job deadline/);

    const replaceRecord = (/** @type {string} */ file, /** @type {unknown} */ value) => {
      fs.unlinkSync(file); fs.writeFileSync(file, JSON.stringify(value) + "\n", {flag: "wx", mode: 0o444});
    };
    replaceRecord(secondState.deadlineRecord, {...second, deadlineEpoch: epoch(46)});
    assert.throws(() => windowsNativeLlvmCiDeadlineRecord(secondState), /replace or reset its original job deadline/);
    replaceRecord(secondRunDeadlineRecord, {version: 1, runDeadlineEpoch: epoch(7 * 60 + 1)});
    assert.throws(() => windowsNativeLlvmCiDeadlineRecord(secondState), /replace or reset its original run deadline/);
  });

  test("runs configure/build/audit phases only through the public builder, not a stage CLI", async () => {
    await assert.rejects(runStage(["windows-native-llvm21", "audit"]), /takes no extra arguments/);
  });
});

test.describe("Windows GPU profile state handoff", () => {
  for (const profile of ["acpp-cuda", "acpp-hip"]) {
    test("factory contract fixtures isolate inherited " + profile + " pins", {skip: process.platform !== "linux"}, () => {
      const env = /** @type {NodeJS.ProcessEnv} */ ({...process.env, MOM_NATIVE_WINDOWS_GPU_PROFILE: profile});
      delete env["NODE_TEST_CONTEXT"]; // Start an independent runner, not a nested test-child protocol.
      for (const name of ["ACPP_SOURCE_BUNDLE", "ACPP_SDK_BUNDLE", "FRONTEND_BUNDLE", "FRONTEND_HANDOFF"]) {
        env["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"] = "f".repeat(64);
      }
      // CI runs all profile contracts inside each factory's real, externally pinned environment.
      const result = spawnSync(process.execPath, ["--test", "--test-reporter=tap", "tests/native-windows-gpu-ci-job.js"], {
        cwd: root, env, encoding: "utf8", timeout: 30000, maxBuffer: 2 * 1024 ** 2,
      });
      assert.equal(result.status, 0, result.error?.message || result.stderr + result.stdout);
      assert.match(result.stdout, /^# tests [1-9]\d*$/m);
      assert.match(result.stdout, /^# skipped 0$/m);
    });
  }

  /** @param {import('node:test').TestContext} t */
  const fixture = t => {
    const runnerTemp = fs.mkdtempSync(path.join(fs.realpathSync("/var/tmp"), "mom-native-win-gpu-handoff-")),
      run = String(Date.now()) + String(process.pid), environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: run + "-1"}), profile = "oneapi";
    environment["MOM_NATIVE_WINDOWS_GPU_DIR"] = path.join(fs.realpathSync(os.tmpdir()), `mom-native-windows-gpu-${profile}-${run}-1`);
    const state = windowsGpuHandoffStatePaths(profile, environment, "linux", "x64");
    fs.mkdirSync(state.isolatedRoot, {recursive: true});
    const config = {version: 1, profile, tools: {clang: "/tools/clang", clangC: "/tools/clang-c", llvmNm: "/tools/llvm-nm",
      llvmReadobj: "/tools/llvm-readobj", llvmObjcopy: "/tools/llvm-objcopy"}, includeRoots: ["/sdk/include"], overlay: "/sdk/overlay.json"};
    fs.writeFileSync(state.configFile, JSON.stringify(config), {flag: "wx"});
    const object = path.join(state.isolatedRoot, "object-0000.obj"), objectBytes = coffObject();
    fs.writeFileSync(object, objectBytes, {flag: "wx"});
    const objectRow = {path: object, bytes: objectBytes.length, sha256: fileHash(object)}, factory = "mom_native_gpu_oneapi",
      graph = {version: 1, profile, factory, configFile: state.configFile, configSha256: fileHash(state.configFile),
        sourceState: nativeSourceInputs(root, "gpu"), sdkInputs: {[state.configFile]: fileHash(state.configFile)}, objects: [objectRow], isolation: state.isolation},
      checks = {nonDirectivePayloadsPreserved: true, nonDirectiveMetadataPreserved: true, comdatSelectionPreserved: true,
        associativeXdataPreserved: true, linkerDirectiveReferencesPreserved: true, externalUndefinedImportsPreserved: true,
        privateUndefinedReferencesRenamed: true, oldCollisionDefinitions: 0, oldCollisionReferences: 0},
      isolation = {version: 1, kind: "windows-coff-graph-isolation", prefix: "mom_oneapi_gpu_", entryPoints: [factory], outputs: [objectRow], checks};
    fs.writeFileSync(state.isolation, JSON.stringify(isolation), {flag: "wx"});
    fs.writeFileSync(state.receipt, JSON.stringify(graph), {flag: "wx"});
    fs.writeFileSync(path.join(state.isolatedRoot, "directives-0.obj"), "diagnostic-only object", {flag: "wx"});
    fs.writeFileSync(path.join(state.isolatedRoot, "directives-0.txt"), "diagnostic-only directives", {flag: "wx"});
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    return {runnerTemp, environment, state, graph, isolation, object};
  };

  test("selects unchanged receipts, config, and only graph-listed objects", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), plan = windowsGpuProfileHandoffPlan("oneapi", f.environment, "linux", "x64", root);
    assert.equal(plan.files.length, 4);
    assert.deepEqual(plan.files.map(row => row.role).sort(), ["factory receipt", "isolation receipt", "isolated object", "profile configuration"].sort());
    assert.ok(!plan.members.some(member => member.includes("directives-")), "isolation diagnostics are not transport inputs");
    assert.equal(plan.files.find(row => row.path === f.object)?.sha256, f.graph.objects[0]?.sha256);
    assert.throws(() => windowsGpuProfileHandoffPlan("dpcpp", f.environment, "linux", "x64", root), /CI work-root plan/);
    assert.throws(() => windowsGpuHandoffStatePaths("invalid", f.environment, "linux", "x64"), /one maintained profile/);
  });

  test("packs and restores an exact selected-file tar while excluding diagnostics", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), packed = packWindowsGpuHandoff("oneapi", f.environment, "linux", "x64", root),
      listing = listState(packed.archive, [f.state.name]).map(name => name.endsWith("/") ? name.slice(0, -1) : name);
    assert.equal(packed.profile, "oneapi"); assert.equal(packed.files, 5); assert.equal(packed.buildOnly, true);
    assert.ok(listing.includes(f.state.name + "/handoff.json"));
    const handoffFile = path.join(f.state.root, "handoff.json"), handoff = JSON.parse(fs.readFileSync(handoffFile, "utf8"));
    assert.equal(packed.handoffSha256, fileHash(handoffFile));
    assert.equal(handoff.format, "mom-windows-gpu-handoff-v1");
    assert.deepEqual(handoff.sdkInputs, f.graph.sdkInputs);
    assert.equal(handoff.files.length, 4);
    assert.ok(listing.includes(f.state.name + "/output/result.json"));
    assert.ok(listing.includes(f.state.name + "/output/isolated/result.json"));
    assert.ok(listing.includes(f.state.name + "/output/isolated/object-0000.obj"));
    assert.ok(listing.includes(f.state.name + "/config.json"));
    assert.ok(!listing.some(name => name.includes("directives-")));
    const restoreBase = fs.mkdtempSync(path.join(f.runnerTemp, "restore-"));
    try {
      extractState(packed.archive, restoreBase, [f.state.name]);
      assert.equal(fileHash(path.join(restoreBase, f.state.name, "output/isolated/object-0000.obj")), f.graph.objects[0]?.sha256);
      assert.equal(fs.existsSync(path.join(restoreBase, f.state.name, "output/isolated/directives-0.obj")), false);
    } finally {fs.rmSync(restoreBase, {recursive: true, force: true});}
    // Exercise the real producer-to-consumer contract at the recorded state path,
    // replacing only this test's disposable producer tree with its compact transport.
    fs.rmSync(f.state.root, {recursive: true});
    extractState(packed.archive, path.dirname(f.state.root), [f.state.name]);
    const admitted = require("../scripts/native-windows-gpu-handoff").windowsGpuHandoffInputs(f.state.receipt,
      {profile: "oneapi", receiptSha256: fileHash(f.state.receipt),
        configSha256: fileHash(f.state.configFile), handoffSha256: packed.handoffSha256});
    assert.equal(admitted.profile, "oneapi");
    assert.equal(Object.keys(admitted.files).length, 5);
  });

  test("rejects a stale or missing producer SDK cohort before creating the handoff archive", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), input = path.join(f.runnerTemp, "sdk-header.h");
    fs.writeFileSync(input, "original SDK header");
    f.graph.sdkInputs = {[input]: fileHash(input)};
    fs.writeFileSync(f.state.receipt, JSON.stringify(f.graph));
    windowsGpuProfileHandoffPlan("oneapi", f.environment, "linux", "x64", root);
    const alias = path.join(f.runnerTemp, "sdk-header-alias.h");
    fs.symlinkSync(input, alias);
    f.graph.sdkInputs = {[alias]: fileHash(alias)};
    fs.writeFileSync(f.state.receipt, JSON.stringify(f.graph));
    windowsGpuProfileHandoffPlan("oneapi", f.environment, "linux", "x64", root);
    fs.appendFileSync(input, " changed");
    assert.throws(() => packWindowsGpuHandoff("oneapi", f.environment, "linux", "x64", root), /SDK input changed before handoff/);
    assert.equal(fs.existsSync(f.state.archive), false);
    f.graph.sdkInputs = {};
    fs.writeFileSync(f.state.receipt, JSON.stringify(f.graph));
    assert.throws(() => packWindowsGpuHandoff("oneapi", f.environment, "linux", "x64", root), /omits its SDK input cohort/);
    assert.equal(fs.existsSync(f.state.archive), false);
  });

  test("shares one deadline across profile admission, tar creation and listing", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), start = Date.now(), timeoutMs = 1000; let calls = 0;
    t.mock.method(Date, "now", () => ++calls <= 2 ? start : start + timeoutMs + 1);
    assert.throws(() => packWindowsGpuHandoff("oneapi", f.environment, "linux", "x64", root, timeoutMs),
      /deadline expired before archive listing/);
    assert.equal(calls, 3, "clock advances only after the archive has been created and before listing starts");
    assert.ok(fs.existsSync(f.state.archive), "the test reaches tar creation but withholds a completed handoff after its shared deadline expires");
  });

  test("rejects altered output paths and supports bounded regular-file selections under eight roots", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), changed = {...f.graph, objects: [{...f.graph.objects[0], path: path.join(f.state.isolatedRoot, "..", "outside.obj")}]};
    fs.writeFileSync(f.state.receipt, JSON.stringify(changed));
    fs.writeFileSync(f.state.isolation, JSON.stringify({...f.isolation, outputs: changed.objects}));
    assert.throws(() => windowsGpuProfileHandoffPlan("oneapi", f.environment, "linux", "x64", root), /exact output directory/);

    const base = path.join(f.runnerTemp, "selected-state"), rootName = "bounded-root", paths = /** @type {string[]} */ ([]);
    fs.mkdirSync(path.join(base, rootName), {recursive: true});
    for (let index = 0; index < 9; index++) {
      const relative = `${rootName}/file-${index}.bin`; fs.writeFileSync(path.join(base, relative), String(index), {flag: "wx"}); paths.push(relative);
    }
    const archive = path.join(f.runnerTemp, "selected-files.tar");
    packState(archive, base, paths); assert.ok(listState(archive, [rootName]).some(name => name === rootName + "/file-8.bin"));
    const roots = /** @type {string[]} */ ([]);
    for (let index = 0; index < 9; index++) {
      const name = "root-" + index; fs.mkdirSync(path.join(base, name)); fs.writeFileSync(path.join(base, name, "one.bin"), "x"); roots.push(name + "/one.bin");
    }
    assert.throws(() => packState(path.join(f.runnerTemp, "too-many-roots.tar"), base, roots), /eight-root limit/);
    if (process.platform !== "win32") {
      fs.symlinkSync(path.join(base, paths[0] || ""), path.join(base, rootName, "linked.bin"));
      assert.throws(() => packState(path.join(f.runnerTemp, "unsafe-file.tar"), base, [...paths.slice(0, 8), rootName + "/linked.bin"]), /explicit regular-file|canonical regular/);
    }
  });

  /** @param {import('node:test').TestContext} t @param {string} [profile] */
  const stageFixture = (t, profile = "oneapi") => {
    const runnerTemp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-native-win-gpu-stage-")),
      run = String(Date.now()) + String(process.pid), attempt = "1", suffix = run + "-" + attempt,
      outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary"),
      incoming = path.join(runnerTemp, "incoming");
    fs.mkdirSync(incoming); fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
    const config = {version: 1, profile, tools: {clang: "/tools/clang", clangC: "/tools/clang-c", llvmNm: "/tools/llvm-nm",
        llvmReadobj: "/tools/llvm-readobj", llvmObjcopy: "/tools/llvm-objcopy"}, includeRoots: ["/sdk/include"], overlay: "/sdk/overlay.json",
      ...(profile === "oneapi" ? {compilerRoot: "/tools/compiler"} : profile === "dpcpp" ? {cuda: {sdk: "/sdk/cuda", images: "/sdk/images"}} : {})},
      configFile = path.join(incoming, "config.json");
    fs.writeFileSync(configFile, JSON.stringify(config) + "\n", {flag: "wx"});
    const environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: `${run}-${attempt}`,
      MOM_NATIVE_WINDOWS_GPU_DIR: path.join(fs.realpathSync(os.tmpdir()), `mom-native-windows-gpu-${profile}-${suffix}`),
      MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 55 * 60 * 1000) / 1000)),
      MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile});
    const state = windowsGpuHandoffStatePaths(profile, environment, "linux", "x64");
    t.after(() => fs.rmSync(state.root, {recursive: true, force: true}));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    return {runnerTemp, environment, state, configFile, configSha256: fileHash(configFile), outputFile, summaryFile};
  };

  test("bounds one profile build and package against its job-start deadline", () => {
    const nowMs = Date.UTC(2026, 10, 8, 12), epoch = (/** @type {number} */ minutes) => String((nowMs + minutes * 60 * 1000) / 1000);
    assert.equal(windowsGpuDeadlineBudget(epoch(55), "build", nowMs).timeoutMs, 45 * 60 * 1000);
    assert.equal(windowsGpuDeadlineBudget(epoch(55), "package", nowMs).timeoutMs, 7 * 60 * 1000);
    assert.throws(() => windowsGpuDeadlineBudget(epoch(12), "build", nowMs), /Insufficient aggregate job time/);
    assert.throws(() => windowsGpuDeadlineBudget(epoch(61), "package", nowMs), /60-minute job bound/);
    assert.throws(() => windowsGpuDeadlineBudget("expired", "build", nowMs), /valid bounded job-start deadline/);
  });

  test("preflights pinned config, profile, root and deadline before any producer/output mutation", {skip: process.platform !== "linux"}, t => {
    const f = stageFixture(t), beforeOutput = fs.readFileSync(f.outputFile, "utf8"), beforeSummary = fs.readFileSync(f.summaryFile, "utf8"),
      nowMs = Date.now(), good = windowsGpuProfileCiPreflight("oneapi", f.configFile, f.configSha256, f.environment, "linux", "x64", nowMs);
    assert.equal(good.config.profile, "oneapi"); assert.equal(good.state.root, f.state.root); assert.ok(good.buildTimeoutMs > 0);
    assert.equal(fs.existsSync(f.state.root), false);
    /** @param {string} profile @param {string} configFile @param {string} sha256 @param {NodeJS.ProcessEnv} environment */
    const rejectsWithoutMutation = (profile, configFile, sha256, environment) => {
      const attemptedRoot = environment["MOM_NATIVE_WINDOWS_GPU_DIR"] || "",
        attemptedArchive = path.join(environment["RUNNER_TEMP"] || "", `native-windows-gpu-${profile}-${environment["MOM_NATIVE_CI_SUFFIX"]}.tar`);
      assert.throws(() => buildWindowsGpuProfileCi(profile, configFile, sha256, environment));
      assert.equal(fs.existsSync(attemptedRoot), false);
      assert.equal(fs.existsSync(path.join(attemptedRoot, "output")), false);
      assert.equal(fs.existsSync(attemptedArchive), false);
      assert.equal(fs.readFileSync(f.outputFile, "utf8"), beforeOutput);
      assert.equal(fs.readFileSync(f.summaryFile, "utf8"), beforeSummary);
    };
    rejectsWithoutMutation("oneapi", f.configFile, "", f.environment);
    rejectsWithoutMutation("oneapi", f.configFile, "f".repeat(64), f.environment);
    const dpcppEnv = {...f.environment, MOM_NATIVE_WINDOWS_GPU_DIR: path.join(fs.realpathSync(os.tmpdir()),
      `mom-native-windows-gpu-dpcpp-${f.state.suffix}`)};
    rejectsWithoutMutation("dpcpp", f.configFile, f.configSha256, dpcppEnv);
    rejectsWithoutMutation("oneapi", f.configFile, f.configSha256,
      {...f.environment, MOM_NATIVE_WINDOWS_GPU_DIR: path.join(f.runnerTemp, "wrong-profile-root")});
    rejectsWithoutMutation("oneapi", f.configFile, f.configSha256,
      {...f.environment, MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH: String(Math.floor(nowMs / 1000) - 1)});
  });

  test("dispatches the explicit Windows GPU stage CLI and copies config before the public producer", async t => {
    // This rejects an invalid CLI profile on either OS; it needs no Linux-only stage root.
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-gpu-cli-")), configFile = path.join(temp, "config.json");
    t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
    fs.writeFileSync(configFile, "{}\n", {flag: "wx"});
    const f = {configFile, configSha256: fileHash(configFile)};
    await assert.rejects(runStage(["windows-native-gpu"]), /explicit profile, config path and config SHA-256/);
    await assert.rejects(runStage(["windows-native-gpu", "invalid", f.configFile, f.configSha256]), /one maintained profile/);
    const cli = spawnSync(process.execPath, [path.join(root, "scripts/run-native-ci.js"), "stage", "windows-native-gpu",
      "invalid", f.configFile, f.configSha256], {encoding: "utf8", timeout: 10_000});
    assert.equal(cli.status, 1); assert.match(cli.stderr || "", /one maintained profile/);
    assert.deepEqual(fs.readdirSync(temp), ["config.json"]);
  });
});

test.describe("Windows controller-support SDK CI", () => {
  /** @param {import('node:test').TestContext} t @param {"io" | "hermes"} [kind] */
  const fixture = (t, kind = "io") => {
    const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-controller-ci-")), suffix = "24680-1",
      workRoot = path.join(runnerTemp, `mom-native-windows-controller-${kind}-${suffix}`),
      outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary");
    fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    const environment = {RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: "24680-1", MOM_BUILD_JOBS: "4",
      [kind === "io" ? "MOM_NATIVE_WINDOWS_CONTROLLER_IO_DIR" : "MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_DIR"]: workRoot,
      MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH: "1790000000", GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile};
    return {runnerTemp, workRoot, outputFile, summaryFile, environment,
      state: windowsControllerStatePaths(environment, "win32", kind)};
  };

  /** @param {string} root */
  const toolFiles = root => {
    const names = ["cmake.exe", "icx-cl.exe", "ninja.exe", "lib.exe", "link.exe", "nmake.exe", "perl.exe", "rc.exe", "tar.exe", "git.exe"],
      tools = Object.create(null);
    for (const name of names) {const file = path.join(root, name); fs.writeFileSync(file, "test tool\n", {flag: "wx"}); tools[name] = file;}
    return tools;
  };

  test("binds both support roles to fresh run-attempt paths", t => {
    const io = fixture(t, "io"), hermes = fixture(t, "hermes");
    assert.equal(io.state.suffix, "24680-1"); assert.equal(io.state.workRoot, io.environment["MOM_NATIVE_WINDOWS_CONTROLLER_IO_DIR"]);
    assert.equal(path.dirname(io.state.outputRoot), io.runnerTemp); assert.equal(path.dirname(io.state.archive), io.runnerTemp);
    assert.equal(hermes.state.hermesSourceTar, path.join(hermes.runnerTemp, "incoming", "native-hermes-state-24680-1-sources.tar"));
    assert.equal(hermes.state.hermesSource, path.join(hermes.state.hermesExtractRoot, hermes.state.hermesSourceName, "roots/hermes"));
    assert.throws(() => windowsControllerStatePaths({...io.environment, MOM_NATIVE_WINDOWS_CONTROLLER_IO_DIR: path.join(io.runnerTemp, "ambient")}, "win32", "io"), /exactly match/);
    assert.throws(() => windowsControllerStatePaths(io.environment, "linux", "io"), /Windows runner/);
  });

  test("requires an exact role-specific absolute regular tool table", t => {
    const f = fixture(t), files = toolFiles(f.runnerTemp), openssl = {ar: files["lib.exe"], cc: files["icx-cl.exe"],
        link: files["link.exe"], make: files["nmake.exe"], perl: files["perl.exe"], rc: files["rc.exe"], tar: files["tar.exe"]},
      ioTools = {cc: files["icx-cl.exe"], cmake: files["cmake.exe"], ninja: files["ninja.exe"], openssl};
    assert.deepEqual(windowsControllerTools({...f.environment, MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify(ioTools)}, "io"), ioTools);
    assert.throws(() => windowsControllerTools({...f.environment,
      MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify({...ioTools, git: files["git.exe"]})}, "io"), /unexpected role or key set/);
    assert.throws(() => windowsControllerTools({...f.environment,
      MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify({...ioTools, openssl: {...openssl, tar: "relative/tar.exe"}})}, "io"), /absolute path/);
    const hermesTools = {git: files["git.exe"], cmake: files["cmake.exe"], cc: files["icx-cl.exe"], cxx: files["icx-cl.exe"], ninja: files["ninja.exe"]};
    assert.deepEqual(windowsControllerTools({...f.environment, MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify(hermesTools)}, "hermes"), hermesTools);
    const missing = path.join(f.runnerTemp, "missing-compiler.exe");
    assert.throws(() => windowsControllerTools({...f.environment,
      MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify({...hermesTools, cc: missing})}, "hermes"),
    error => error instanceof Error && error.message.includes("missing, redirected or non-regular: " + missing));
  });

  test("requires resolved tool identity rather than a mutable SDK directory alias", t => {
    const f = fixture(t, "hermes"), real = path.join(f.runnerTemp, "versioned-tools"), alias = path.join(f.runnerTemp, "latest");
    fs.mkdirSync(real);
    const files = toolFiles(real), tools = {git: files["git.exe"], cmake: files["cmake.exe"], cc: files["icx-cl.exe"],
      cxx: files["icx-cl.exe"], ninja: files["ninja.exe"]};
    fs.symlinkSync(real, alias, process.platform === "win32" ? "junction" : "dir");
    const redirected = path.join(alias, "icx-cl.exe");
    assert.throws(() => windowsControllerTools({...f.environment,
      MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify({...tools, cc: redirected})}, "hermes"),
    error => error instanceof Error && error.message.includes("missing, redirected or non-regular: " + redirected));
    assert.deepEqual(windowsControllerTools({...f.environment,
      MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS: JSON.stringify({...tools, cc: fs.realpathSync(redirected)})}, "hermes"), tools);
  });

  test("uses only the public Windows I/O/OpenSSL and Hermes producer frontdoors", t => {
    const io = fixture(t, "io"), hermes = fixture(t, "hermes"), files = toolFiles(io.runnerTemp),
      ioTools = {cc: files["icx-cl.exe"], cmake: files["cmake.exe"], ninja: files["ninja.exe"], openssl: {
        ar: files["lib.exe"], cc: files["icx-cl.exe"], link: files["link.exe"], make: files["nmake.exe"],
        perl: files["perl.exe"], rc: files["rc.exe"], tar: files["tar.exe"]}},
      hermesTools = {git: files["git.exe"], cmake: files["cmake.exe"], cc: files["icx-cl.exe"], cxx: files["icx-cl.exe"], ninja: files["ninja.exe"]},
      ioCommands = windowsControllerIoBuildCommands(io.state, ioTools), hermesCommand = windowsControllerHermesBuildCommand(hermes.state, hermesTools);
    assert.deepEqual(ioCommands.map(row => row.script), ["scripts/build-native-io.js", "scripts/build-native-openssl.js"]);
    assert.deepEqual(ioCommands[0]?.args.slice(-4), [files["cmake.exe"], files["icx-cl.exe"], files["ninja.exe"], "4"]);
    assert.equal(ioCommands[1]?.args[2], io.state.opensslToolsFile);
    assert.equal(hermesCommand.script, "scripts/build-native-hermes.js");
    assert.deepEqual(hermesCommand.args.slice(2), [files["git.exe"], files["cmake.exe"], files["icx-cl.exe"], files["icx-cl.exe"], files["ninja.exe"], "4"]);
    assert.equal(hermesCommand.args[0], hermes.state.hermesSource);
  });

  test("selects the pinned Intel 2026.2 runtime libraries and accompanying license files", () => {
    const compiler = path.join(os.tmpdir(), "Intel oneAPI", "compiler", windowsIntelRuntimeCatalog.compiler.version, "bin", "icx-cl.exe"),
      entries = windowsControllerIntelRuntimeEntries(compiler);
    assert.equal(windowsIntelRuntimeCatalog.version, 1);
    assert.match(windowsIntelRuntimeCatalog.compiler.version, /^\d{4}\.\d+(?:\.\d+)?$/);
    assert.match(windowsIntelRuntimeCatalog.compiler.sha256, /^[a-f0-9]{64}$/);
    assert.deepEqual(windowsIntelRuntimeCatalog.libraries.map(row => row.name),
      ["libircmt.lib", "libirc.lib", "libmmt.lib", "svml_dispmt.lib", "libdecimal.lib"]);
    assert.equal(windowsIntelRuntimeCatalog.licenses.length, 3);
    const expected = [
      ...windowsIntelRuntimeCatalog.libraries.map(row => ({...row, kind: "library"})),
      ...windowsIntelRuntimeCatalog.licenses.map(row => ({...row, kind: "license"})),
    ];
    assert.deepEqual(entries, expected.map(row => ({...row, source: path.join(path.dirname(path.dirname(compiler)), ...row.source.split("/"))})));
    assert.ok(entries.every(row => row.source.startsWith(path.join(os.tmpdir(), "Intel oneAPI", "compiler", windowsIntelRuntimeCatalog.compiler.version))));
    assert.equal(entries.reduce((sum, row) => sum + row.bytes, 0),
      [...windowsIntelRuntimeCatalog.libraries, ...windowsIntelRuntimeCatalog.licenses].reduce((sum, row) => sum + row.bytes, 0));
    assert.ok(entries.reduce((sum, row) => sum + row.bytes, 0) < (512 << 20));
    assert.throws(() => windowsControllerIntelRuntimeEntries("relative/compiler/" + windowsIntelRuntimeCatalog.compiler.version + "/bin/icx-cl.exe"), /selected catalog/);
    assert.throws(() => windowsControllerIntelRuntimeEntries(path.join(os.tmpdir(), "compiler", "other-version", "bin", "icx-cl.exe")), /selected catalog/);
  });

  test("binds copied Hermes root license and source headers to the unchanged producer source map", () => {
    const sha256 = "a".repeat(64), licenseSha256 = "d".repeat(64), files = [
      {path: "hermes/source/include/hermes/RuntimeConfig.h", bytes: 12, sha256},
      {path: "hermes/source/LICENSE", bytes: 1088, sha256: licenseSha256},
      {path: "hermes/build/lib/config/BuildConfig.h", bytes: 7, sha256: "b".repeat(64)},
    ];
    assert.equal(verifyWindowsHermesSourceHandoff(files, {"include/hermes/RuntimeConfig.h": sha256, LICENSE: licenseSha256}), 2);
    assert.throws(() => verifyWindowsHermesSourceHandoff(files, {}), /tracked-source hashes/);
    assert.throws(() => verifyWindowsHermesSourceHandoff(files, {"include/hermes/RuntimeConfig.h": sha256, LICENSE: "e".repeat(64)}), /tracked-source hashes/);
  });

  test("budgets each phase to the absolute job deadline with package time left", () => {
    const now = Date.UTC(2026, 10, 8, 12);
    /** @param {number} minutes */
    const epoch = minutes => String((now + minutes * 60 * 1000) / 1000);
    assert.deepEqual(windowsControllerDeadlineBudget(epoch(55), "fetch", now), {timeoutMs: 10 * 60 * 1000, remainingMs: 55 * 60 * 1000});
    assert.equal(windowsControllerDeadlineBudget(epoch(55), "extract", now).timeoutMs, 5 * 60 * 1000);
    assert.equal(windowsControllerDeadlineBudget(epoch(55), "build", now).timeoutMs, 45 * 60 * 1000);
    assert.equal(windowsControllerDeadlineBudget(epoch(55), "package", now).timeoutMs, 5 * 60 * 1000);
    assert.throws(() => windowsControllerDeadlineBudget("bad", "fetch", now), /job-start deadline/);
    assert.throws(() => windowsControllerDeadlineBudget(epoch(61), "build", now), /60-minute job bound/);
    assert.throws(() => windowsControllerDeadlineBudget(epoch(2), "build", now), /Insufficient aggregate job time/);
  });

  test("packages only the selected regular-file handoff and rejects redirected inputs", t => {
    const f = fixture(t), tarTool = process.platform === "win32" ? path.join(process.env["SystemRoot"] || "C:\\Windows", "System32", "tar.exe") : "/usr/bin/tar";
    fs.mkdirSync(f.state.outputRoot, {mode: 0o700});
    fs.mkdirSync(path.join(f.state.outputRoot, "io-sdk/include"), {recursive: true});
    fs.writeFileSync(path.join(f.state.outputRoot, "io-sdk/include/uv.h"), "pinned header\n", {flag: "wx"});
    const inventory = windowsControllerPackageInventory(f.state.outputRoot, f.state.outputName);
    assert.equal(inventory.files.length, 1); assert.equal(inventory.files[0]?.path, "io-sdk/include/uv.h");
    assert.equal(inventory.bytes, Buffer.byteLength("pinned header\n"));
    const result = packWindowsControllerHandoff(f.state, tarTool, {kind: "io", buildOnly: true}, 30_000);
    assert.equal(result.files, 1); assert.equal(result.selectedBytes, inventory.bytes);
    const handoff = JSON.parse(fs.readFileSync(path.join(f.state.outputRoot, "handoff.json"), "utf8"));
    assert.equal(handoff.bundleSha256, undefined); assert.deepEqual(handoff.files, inventory.files);
    assert.equal(fileHash(f.state.archive), result.bundleSha256);
    const listing = spawnSync(tarTool, ["-tf", f.state.archive], {encoding: "utf8", timeout: 5000});
    assert.equal(listing.status, 0, listing.stderr);
    assert.ok(listing.stdout.split(/\r?\n/).filter(Boolean).some(name => name.replaceAll("\\", "/").endsWith("/io-sdk/include/uv.h")));
  });
});
