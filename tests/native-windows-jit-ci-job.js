"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {windowsJitCiJobPlan, validateWindowsJitConfig, composeWindowsJitConfig, writeWindowsJitConfig,
  validateWindowsJitDescriptorInputs} = require("../scripts/run-native-windows-jit-ci");
const {validateWindowsJitDescriptor, windowsJitToolchainPlan} = require("../scripts/prepare-native-jit-toolchain");

const hash = "a".repeat(64), now = Date.now();

/** @param {string} root @param {string} stage */
function environment(root, stage = "configure") {
  const suffix = "ci-9000000000000008-88", runnerTemp = path.join(root, "runner"), future = Math.floor(now / 1000),
    env = /** @type {NodeJS.ProcessEnv} */ ({
      MOM_NATIVE_CI_SUFFIX: "ci-9000000000000008-88", GITHUB_JOB: "windows-jit",
      RUNNER_TEMP: runnerTemp, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
      GITHUB_OUTPUT: path.join(root, "outputs"), GITHUB_STEP_SUMMARY: path.join(root, "summary"),
      MOM_NATIVE_LINUX_SOURCES_DIR: path.join("/tmp", "mom-native-linux-source-prerequisites-" + suffix),
      MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: path.join("/tmp", "mom-native-dpcpp-cuda-inputs-" + suffix),
      MOM_NATIVE_WINDOWS_JIT_DIR: path.join("/tmp", "mom-native-windows-jit-" + suffix),
      MOM_NATIVE_WINDOWS_JIT_RESOURCES_DIR: path.join("/tmp", "mom-native-windows-jit-resources-" + suffix),
      MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_FILE: path.join(runnerTemp, "mom-native-windows-llvm21-descriptor-" + suffix, "descriptor.json"),
      MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_ROOT: path.join(runnerTemp, "mom-native-windows-host-sdk-" + suffix, "native-windows-host-sdk-" + suffix),
      MOM_NATIVE_WINDOWS_JIT_DEADLINE_EPOCH: String(future + 50 * 60),
      MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH: String(future + 7 * 60 * 60),
      MOM_NATIVE_WINDOWS_JIT_SOURCE_BUNDLE_SHA256: hash, MOM_NATIVE_WINDOWS_JIT_DPCPP_BUNDLE_SHA256: hash,
      MOM_NATIVE_WINDOWS_JIT_DPCPP_MANIFEST_SHA256: hash,
      MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256: hash, MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256: hash,
      MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_BUNDLE_SHA256: hash, MOM_NATIVE_WINDOWS_CONTROLLER_HERMES_HANDOFF_SHA256: hash,
      MOM_NATIVE_WINDOWS_JIT_RESOURCES_BUNDLE_SHA256: hash, MOM_NATIVE_WINDOWS_JIT_DESCRIPTOR_SHA256: hash,
      MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_RESULT_SHA256: hash, MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_DESCRIPTOR_SHA256: hash,
      MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_FILES_SHA256: hash, MOM_NATIVE_WINDOWS_JIT_WINDOWS_SDK_OVERLAY_SHA256: hash,
    });
  if (stage !== "configure") {
    const previous = stage === "support" ? "configured" : ({targets: "support", frontend: "targets", link: "frontend"})[stage];
    env["MOM_NATIVE_WINDOWS_JIT_STATE_PREVIOUS_PHASE"] = previous;
    env["MOM_NATIVE_WINDOWS_JIT_STATE_BUNDLE_SHA256"] = hash;
  }
  fs.mkdirSync(runnerTemp, {recursive: true});
  return env;
}

/** @param {string} root @param {string} receiptFile */
function nativeTablegenInputs(root, receiptFile) {
  const toolRoot = path.join(root, "resources", "native-tools");
  return {toolRoot, manifestFile: path.join(toolRoot, "manifest.json"), manifestSha256: hash,
    files: {clangTableGen: path.join(toolRoot, "bin", "clang-tblgen"), llvmTableGen: path.join(toolRoot, "bin", "llvm-tblgen")},
    hashes: {clangTableGenSha256: hash, llvmTableGenSha256: hash},
    source: {receiptFile, receiptSha256: hash, commit: "b".repeat(40)}};
}

/** Map the production's pinned tool paths to real executable fixture files on any Linux host.
 * @param {import("node:test").TestContext} t @param {string} root */
function mockCanonicalTools(t, root) {
  const directory = path.join(root, "canonical-tools"); fs.mkdirSync(directory, {recursive: true});
  const tools = {cmake: path.join(directory, "cmake"), ninja: path.join(directory, "ninja")};
  for (const file of Object.values(tools)) {
    fs.writeFileSync(file, "#!/bin/sh\n"); fs.chmodSync(file, 0o755); fs.accessSync(file, fs.constants.X_OK);
  }
  const canonical = {cmake: fs.realpathSync(tools.cmake), ninja: fs.realpathSync(tools.ninja)}, realpathSync = fs.realpathSync.bind(fs);
  t.mock.method(fs, "realpathSync", /** @type {typeof fs.realpathSync} */ ((file) => {
    if (String(file) === "/usr/bin/cmake") {return canonical.cmake;}
    if (String(file) === "/usr/bin/ninja") {return canonical.ninja;}
    return realpathSync(file);
  }));
  return canonical;
}

test.describe("Windows original source-JIT CI phases", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @type {string} */ let root;
  test.beforeEach(() => {root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-ci-"));});
  test.afterEach(() => {fs.rmSync(root, {recursive: true, force: true});});

  test("uses the original four phase sequence and same-run state names", () => {
    const env = environment(root), config = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json"),
      plan = windowsJitCiJobPlan("configure", config, undefined, env, "linux", "x64", now);
    assert.equal(plan.jobId, "windows-jit_configure");
    assert.equal(plan.root, "/tmp/mom-native-windows-jit-ci-9000000000000008-88");
    assert.equal(plan.descriptor, path.join(env["RUNNER_TEMP"] || "", "mom-native-windows-llvm21-descriptor-ci-9000000000000008-88", "descriptor.json"));
    assert.equal(plan.resourcesArtifactRoot, "/tmp/mom-native-windows-jit-resources-ci-9000000000000008-88");
    assert.equal(plan.resourcesRoot, path.join(plan.resourcesArtifactRoot, "resources"));
    assert.equal(plan.dpcppName, "mom-native-dpcpp-cuda-inputs-ci-9000000000000008-88");
    assert.equal(path.basename(plan.dpcppArchive), "native-linux-dpcpp-cuda-inputs-ci-9000000000000008-88.tar");
    assert.equal(plan.archive, path.join(env["RUNNER_TEMP"] || "", "native-windows-jit-sdk-ci-9000000000000008-88-configured.tar"));
    assert.equal(plan.previousPhase, null);
    for (const stage of ["support", "targets", "frontend", "link"]) {
      const stagedEnv = environment(root, stage), stagedConfig = path.join(stagedEnv["RUNNER_TEMP"] || "",
          "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json"), staged = windowsJitCiJobPlan(stage, stagedConfig, hash, stagedEnv, "linux", "x64", now);
      assert.equal(staged.previousPhase, ({support: "configured", targets: "support", frontend: "targets", link: "frontend"})[stage]);
      assert.equal(staged.jobId, "windows-jit_" + stage);
    }
  });

  test("mints one bounded run deadline only for configure", () => {
    const env = environment(root); delete env["MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH"];
    const config = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json"),
      plan = windowsJitCiJobPlan("configure", config, undefined, env, "linux", "x64", now);
    assert.equal(Number(plan.runDeadlineEpoch), Math.floor(now / 1000) + 8 * 60 * 60);
    assert.throws(() => windowsJitCiJobPlan("support", config, hash, {...env, GITHUB_JOB: "windows-jit"}, "linux", "x64", now), /deadline/);
  });

  test("rejects unknown phases, malformed identities and non-Linux or non-build-only jobs", () => {
    const env = environment(root), config = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json");
    for (const stage of ["all", "configure;link", "constructor"]) {assert.throws(() => windowsJitCiJobPlan(stage, config, hash, env, "linux", "x64", now));}
    for (const changed of [{...env, MOM_NATIVE_CI_SUFFIX: "../x"}, {...env, MOM_NATIVE_CI_SUFFIX: ""}, {...env, MOM_BUILD_JOBS: "3"}]) {
      assert.throws(() => windowsJitCiJobPlan("configure", config, undefined, changed, "linux", "x64", now));
    }
    assert.throws(() => windowsJitCiJobPlan("configure", config, undefined, env, "win32", "x64", now));
  });

  test("rejects stale config paths, state predecessors and absent external pins", () => {
    const env = environment(root), config = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json");
    assert.throws(() => windowsJitCiJobPlan("configure", "/tmp/stale.json", undefined, env, "linux", "x64", now), /paths/);
    assert.throws(() => windowsJitCiJobPlan("targets", config, hash, {...env, MOM_NATIVE_WINDOWS_JIT_STATE_PREVIOUS_PHASE: "configured"}, "linux", "x64", now), /paths/);
    assert.throws(() => windowsJitCiJobPlan("configure", config, undefined, {...env, MOM_NATIVE_WINDOWS_JIT_RESOURCES_BUNDLE_SHA256: ""}, "linux", "x64", now), /external/);
    assert.throws(() => windowsJitCiJobPlan("configure", config, undefined, {...env, MOM_NATIVE_WINDOWS_JIT_DPCPP_MANIFEST_SHA256: ""}, "linux", "x64", now), /external/);
  });

  test("limits each phase to 55 minutes and never extends the original eight-hour run", () => {
    const env = environment(root), config = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json");
    assert.throws(() => windowsJitCiJobPlan("configure", config, undefined,
      {...env, MOM_NATIVE_WINDOWS_JIT_DEADLINE_EPOCH: String(Math.floor(now / 1000) + 56 * 60)}, "linux", "x64", now), /55-minute/);
    assert.throws(() => windowsJitCiJobPlan("configure", config, undefined,
      {...env, MOM_NATIVE_WINDOWS_JIT_RUN_DEADLINE_EPOCH: String(Math.floor(now / 1000) + 9 * 60 * 60)}, "linux", "x64", now), /8-hour/);
  });

  test("composes the original config from admitted same-run inputs and writes it once", t => {
    mockCanonicalTools(t, root);
    const env = environment(root), configPath = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json"),
      plan = windowsJitCiJobPlan("configure", configPath, undefined, env, "linux", "x64", now),
      compilerReceipt = path.join(root, "linux", "compiler.json"), nativeJitTools = nativeTablegenInputs(root, compilerReceipt),
      admitted = {linux: {compilerPreparedReceipt: compilerReceipt},
        dpcpp: {dpcppRoot: path.join(root, "dpcpp")}, compiler: path.join(root, "intel", "icx-cl"),
        nativeJitTools,
        toolchainResult: {toolchain: path.join(plan.toolchain, "toolchain.cmake"),
          inputs: {[path.join(root, "host-sdk", "vfs-overlay.json")]: hash, [path.join(plan.toolchain, "toolchain.cmake")]: hash,
            [nativeJitTools.files.clangTableGen]: hash, [nativeJitTools.files.llvmTableGen]: hash}}},
      config = composeWindowsJitConfig(plan, admitted);
    assert.equal(config.target, "windows"); assert.equal(config.jobs, 2);
    assert.equal(config.sources, admitted.linux.compilerPreparedReceipt); assert.equal(config.sdk, admitted.dpcpp.dpcppRoot);
    assert.equal(config.resources, plan.resourcesRoot); assert.equal(config.tools.cc, admitted.compiler); assert.equal(config.tools.cxx, admitted.compiler);
    assert.equal(config.tools.toolchain, admitted.toolchainResult.toolchain);
    assert.deepEqual(config.inputs, [...new Set([...Object.keys(admitted.toolchainResult.inputs), nativeJitTools.manifestFile,
      compilerReceipt, nativeJitTools.files.clangTableGen, nativeJitTools.files.llvmTableGen, plan.descriptor, admitted.compiler,
      fs.realpathSync("/usr/bin/cmake"), fs.realpathSync("/usr/bin/ninja")])].sort());
    assert.ok(config.inputs.includes(nativeJitTools.manifestFile));
    assert.ok(config.inputs.includes(nativeJitTools.files.clangTableGen));
    assert.ok(config.inputs.includes(nativeJitTools.files.llvmTableGen));
    const written = writeWindowsJitConfig(plan, config);
    assert.equal(written.sha256, require("../scripts/build-source-inputs").fileHash(configPath));
    assert.deepEqual(written.value, config);
    assert.throws(() => writeWindowsJitConfig(plan, config), /fresh/);
    const supportPlan = windowsJitCiJobPlan("support", configPath, hash, environment(root, "support"), "linux", "x64", now);
    assert.deepEqual(composeWindowsJitConfig(supportPlan, admitted), config);
    assert.throws(() => composeWindowsJitConfig(plan, {...admitted, nativeJitTools: undefined}), /TableGen admission/);
    assert.throws(() => composeWindowsJitConfig(plan, {...admitted,
      nativeJitTools: {...nativeJitTools, source: {...nativeJitTools.source, receiptFile: path.join(root, "other-sources.json")}}}), /same-run inputs/);
    assert.throws(() => composeWindowsJitConfig(plan, {...admitted,
      toolchainResult: {...admitted.toolchainResult, inputs: {...admitted.toolchainResult.inputs,
        [nativeJitTools.files.clangTableGen]: "f".repeat(64)}}}), /same-run inputs/);
    assert.throws(() => composeWindowsJitConfig(plan, {...admitted,
      nativeJitTools: {...nativeJitTools, files: {...nativeJitTools.files, llvmTableGen: path.join(root, "outside-tblgen")}}}), /TableGen admission/);
  });

  test("requires a configure-generated config to be fresh and prevents report aliasing", () => {
    const env = environment(root), configPath = path.join(env["RUNNER_TEMP"] || "", "incoming/config/native-windows-jit-sdk-ci-9000000000000008-88.json");
    assert.throws(() => windowsJitCiJobPlan("configure", configPath, hash, env, "linux", "x64", now));
    assert.throws(() => windowsJitCiJobPlan("configure", configPath, undefined, {...env, GITHUB_OUTPUT: configPath}, "linux", "x64", now), /report files/);
  });

  test("admits only the original Windows builder schema and explicit current roots", t => {
    const tools = mockCanonicalTools(t, root);
    const runRoot = "/tmp/mom-jit-run", config = {version: 1, target: "windows", sources: runRoot + "/sources.json", sdk: runRoot + "/dpcpp",
        resources: runRoot + "/resources", tools: {cc: runRoot + "/icx-real", cxx: runRoot + "/icx-real",
          cmake: tools.cmake, ninja: tools.ninja, toolchain: runRoot + "/toolchain/toolchain.cmake"},
        jobs: 2, inputs: [runRoot + "/toolchain/toolchain.cmake", runRoot + "/descriptor.json", runRoot + "/icx-real"]},
      expected = {sources: config.sources, sdk: config.sdk, resources: config.resources, toolchain: config.tools.toolchain,
        compiler: config.tools.cc, jobs: 2, requiredInputs: [config.tools.toolchain, runRoot + "/descriptor.json", runRoot + "/icx-real"], allowedRoots: [runRoot]};
    assert.deepEqual(validateWindowsJitConfig(config, expected), config);
    for (const changed of [
      {...config, target: "linux"}, {...config, jobs: 4}, {...config, tools: {...config.tools, cc: "/usr/bin/clang"}},
      {...config, resources: "/tmp/old-resources"}, {...config, inputs: [config.tools.toolchain]},
      {...config, inputs: [...config.inputs, runRoot + "/../private/libfoo.lib"]},
      {...config, inputs: [...config.inputs, runRoot + "/descriptor.json"]}, {...config, tools: {...config.tools, cmake: path.join(runRoot, "missing-cmake")}},
      {...config, tools: {...config.tools, ninja: path.join(runRoot, "missing-ninja")}}, {...config, unsupported: true},
    ]) {assert.throws(() => validateWindowsJitConfig(changed, expected));}
  });

  test("runner consumes the partial LLVM descriptor without requiring controller Hermes fields", () => {
    const descriptor = {version: 1,
        tools: {clangCl: "/opt/intel/icx-cl", lldLink: "/sdk/bin/lld-link", llvmNm: "/sdk/bin/llvm-nm", clangResourceInclude: "/opt/intel/include"},
        windowsSdk: {overlay: "/sdk/vfs.json", sourceRoot: "/sdk", includeRoots: ["/sdk/include"], libraryRoots: ["/sdk/lib"],
          msvcLibraries: ["/sdk/lib/libcmt.lib"], ucrtLibraries: ["/sdk/lib/ucrt/libucrt.lib"], importLibraries: ["/sdk/lib/um/kernel32.lib"]}},
      plan = windowsJitToolchainPlan(descriptor, "/dpcpp", "/resources/native-tools/bin", "/new-toolchain"),
      runner = fs.readFileSync(path.join(__dirname, "../scripts/run-native-windows-jit-ci.js"), "utf8");
    assert.equal(plan.config.windowsSdk.overlay, descriptor.windowsSdk.overlay);
    assert.equal(plan.compilerTools.LLVM_TABLEGEN, "/resources/native-tools/bin/llvm-tblgen");
    assert.equal(plan.compilerTools.CLANG_TABLEGEN, "/resources/native-tools/bin/clang-tblgen");
    assert.deepEqual(validateWindowsJitDescriptor(descriptor), descriptor);
    assert.match(runner, /validateWindowsJitDescriptor\(readBuildJson\(plan\.descriptor\)\)/);
    assert.doesNotMatch(runner, /descriptor\.hermes\.(?:source|build)/);
    assert.throws(() => windowsJitToolchainPlan({...descriptor, tools: {...descriptor.tools, llvmNm: undefined}},
      "/dpcpp", "/resources/native-tools/bin", "/new-toolchain"));
    assert.throws(() => windowsJitToolchainPlan({...descriptor, windowsSdk: {...descriptor.windowsSdk, importLibraries: []}},
      "/dpcpp", "/resources/native-tools/bin", "/new-toolchain"));
  });

  test("admit the current same-source native TableGen handoff while retaining thin LLVM21 utilities", () => {
    const runner = fs.readFileSync(path.join(__dirname, "../scripts/run-native-windows-jit-ci.js"), "utf8");
    assert.match(runner, /admitWindowsJitNativeTools\(resourcesArtifactRoot,\s*plan\.suffix\)/);
    assert.match(runner, /prepareWindowsJitToolchain\(plan\.descriptor,\s*dpcpp\.dpcppRoot,\s*path\.join\(nativeJitTools\.toolRoot,\s*"bin"\)/);
    assert.match(runner, /toolchainResult\.inputs\[nativeJitTools\.files\.clangTableGen\] !== nativeJitTools\.hashes\.clangTableGenSha256/);
    assert.match(runner, /toolchainResult\.inputs\[nativeJitTools\.files\.llvmTableGen\] !== nativeJitTools\.hashes\.llvmTableGenSha256/);
    assert.match(runner, /linuxLlvmToolHandoffInputs\(toolsRoot/);
    assert.match(runner, /validateWindowsJitDescriptorInputs\(descriptor, \{sdkRoot: host\.sdkRoot, sdk: host\.sdk, tools, intelRuntime\}\)/);
  });

  test("admits the partial LLVM descriptor with exact SDK roots plus the Hermes Intel runtime root", () => {
    const sdk = {sourceRoot: "/sdk", overlay: "/sdk/vfs-overlay.json", includeRoots: ["/sdk/include/shared", "/sdk/include/ucrt"],
        libraryRoots: ["/sdk/sdk/lib/ucrt", "/sdk/sdk/lib/um", "/sdk/msvc/lib"],
        msvcLibraries: ["/sdk/msvc/lib/libcmt.lib", "/handoff/intel-runtime/lib/libircmt.lib", "/handoff/intel-runtime/lib/libmmt.lib"],
        ucrtLibraries: ["/sdk/sdk/lib/ucrt/libucrt.lib"], importLibraries: ["/sdk/sdk/lib/um/kernel32.lib"]},
      intelRuntime = {libRoot: "/handoff/intel-runtime/lib", libraries: {
        libircmt: "/handoff/intel-runtime/lib/libircmt.lib", libmmt: "/handoff/intel-runtime/lib/libmmt.lib",
      }}, tools = {lldLink: "/llvm/bin/lld-link", llvmNm: "/llvm/bin/llvm-nm"},
      host = {sdkRoot: sdk.sourceRoot, sdk}, descriptor = {version: 1,
        tools: {clangCl: "/intel/bin/icx-cl", lldLink: tools.lldLink, llvmNm: tools.llvmNm, clangResourceInclude: "/intel/include"},
        windowsSdk: {...sdk, libraryRoots: [...sdk.libraryRoots, intelRuntime.libRoot]}},
      expected = {sdkRoot: "/sdk", sdk, tools, intelRuntime};

    assert.equal(validateWindowsJitDescriptorInputs(descriptor, {...host, tools, intelRuntime}), true);
    for (const libraryRoots of [
      sdk.libraryRoots,
      ["/sdk/sdk/lib/um", "/sdk/sdk/lib/ucrt", "/sdk/msvc/lib", intelRuntime.libRoot],
      [...sdk.libraryRoots, "/sdk/extra/lib", intelRuntime.libRoot],
    ]) {
      assert.throws(() => validateWindowsJitDescriptorInputs({...descriptor, windowsSdk: {...descriptor.windowsSdk, libraryRoots}}, expected), /descriptor differs/);
    }
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected,
      intelRuntime: {...intelRuntime, libRoot: "/handoff/wrong/lib"}}), /descriptor differs/);
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected,
      intelRuntime: {...intelRuntime, libraries: {...intelRuntime.libraries, libircmt: "/outside/libircmt.lib"}}}), /descriptor differs/);
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected, sdkRoot: "/wrong/sdk"}), /descriptor differs/);
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected,
      sdk: {...sdk, overlay: "/wrong/vfs-overlay.json"}}), /descriptor differs/);
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected,
      tools: {...tools, lldLink: "/wrong/lld-link"}}), /descriptor differs/);
    assert.throws(() => validateWindowsJitDescriptorInputs(descriptor, {...expected,
      tools: {...tools, llvmNm: "/wrong/llvm-nm"}}), /descriptor differs/);
  });

  test("requires the canonical compiler path after symlink resolution", t => {
    mockCanonicalTools(t, root);
    const runRoot = path.join(root, "intel"), compiler = path.join(runRoot, "icx-real"), alias = path.join(runRoot, "icx-cl");
    fs.mkdirSync(runRoot); fs.writeFileSync(compiler, "test compiler"); fs.symlinkSync(compiler, alias);
    const canonical = fs.realpathSync(alias), config = {version: 1, target: "windows", sources: runRoot + "/sources.json", sdk: runRoot + "/dpcpp",
        resources: runRoot + "/resources", tools: {cc: canonical, cxx: canonical,
          cmake: fs.realpathSync("/usr/bin/cmake"), ninja: fs.realpathSync("/usr/bin/ninja"), toolchain: runRoot + "/toolchain.cmake"},
        jobs: 2, inputs: [runRoot + "/toolchain.cmake", runRoot + "/descriptor.json", canonical]},
      expected = {sources: config.sources, sdk: config.sdk, resources: config.resources, toolchain: config.tools.toolchain,
        compiler: canonical, jobs: 2, requiredInputs: config.inputs, allowedRoots: [runRoot]};
    assert.deepEqual(validateWindowsJitConfig(config, expected), config);
    assert.throws(() => validateWindowsJitConfig({...config, tools: {...config.tools, cc: alias, cxx: alias}}, expected), /admitted same-run/);
  });
});
