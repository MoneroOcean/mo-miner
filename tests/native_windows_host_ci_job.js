"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"),
  {spawnSync} = require("node:child_process"), {test} = require("node:test");

const repo = path.resolve(__dirname, ".."), workflowPath = path.join(repo, ".github/workflows/build-windows-host-job.yml"),
  callerPath = path.join(repo, ".github/workflows/build-native-prerequisites.yml"),
  workflow = fs.readFileSync(workflowPath, "utf8"), caller = fs.readFileSync(callerPath, "utf8");
const {clrRevision, sourceFiles: hiprtcSourceFiles} = require("../scripts/build-native-windows-hiprtc-sdk");

/** @param {string} name */
function step(name, source = workflow) {
  const lines = source.split(/\r?\n/), start = lines.findIndex(line => line === "      - name: " + name);
  assert.notEqual(start, -1, "missing workflow step: " + name);
  let end = start + 1;
  while (end < lines.length && !/^ {6}- name: /.test(lines[end] || "")) {end++;}
  return lines.slice(start, end).join("\n");
}

/** @param {string} block */
function runBody(block) {
  const match = block.match(/^ {8}run: \|\r?\n/m);
  if (!match || match.index === undefined) {throw new Error("step needs a literal run block");}
  return block.slice(match.index + match[0].length).split(/\r?\n/)
    .map((/** @type {string} */ line) => line.trim() ? line.replace(/^ {10}/, "") : "").join("\n").replace(/\n+$/, "");
}

/** @param {string} block */
function runCommand(block) {
  const inline = block.match(/^ {8}run: (\S.*)$/m);
  return inline?.[1] || runBody(block);
}

/** @param {string} source @param {NodeJS.ProcessEnv} env */
function runBash(source, env) {
  return spawnSync("/bin/bash", ["-e", "-u", "-o", "pipefail", "-c", source], {cwd: repo, env, encoding: "utf8", timeout: 10_000});
}

/** @param {string[]} args @param {NodeJS.ProcessEnv} [environment] */
function runGit(args, environment = {}) {
  const result = spawnSync("/usr/bin/git", args, {cwd: repo, env: {...process.env, ...environment}, encoding: "utf8", timeout: 10_000});
  assert.equal(result.status, 0, result.stderr);
  return (result.stdout || "").trim();
}

/** @param {import("node:test").TestContext} t */
function tempDir(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-windows-host-job-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  return directory;
}

/** @param {string} root */
function hostComponentFixture(root) {
  const {archiveSha256} = require("../scripts/build-native-windows-llvm21-sdk"),
    {revision: comgrRevision, hostSources} = require("../scripts/build-native-windows-comgr-sdk"),
    {fileHash} = require("../scripts/build-source-inputs"), acppRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6",
    pins = {version: 3, targetTriple: "x86_64-pc-windows-msvc", linkOrder: ["acpp", "hiprtc", "comgr", "llvm"],
      llvm: {sourceArchiveSha256: archiveSha256, archives: 164},
      comgr: {sourceRevision: comgrRevision, archives: 1, objects: hostSources.length},
      acpp: {sourceRevision: acppRevision, archives: 9},
      hiprtc: {sourceRevision: clrRevision, comgrRevision, archiveName: "hiprtc-host-intel.lib", archives: 1, objects: hiprtcSourceFiles.length + 1}},
    groups = {acpp: 9, hiprtc: 1, comgr: 1, llvm: 164}, generationId = "123e4567-e89b-42d3-a456-426614174000",
    link = /** @type {Record<"acpp" | "hiprtc" | "comgr" | "llvm", string[]>} */ ({}),
    files = /** @type {Record<string, {sha256: string, bytes: number}>} */ ({});
  fs.mkdirSync(root);
  for (const [group, count] of /** @type {Array<[keyof typeof groups, number]>} */ (Object.entries(groups))) {
    fs.mkdirSync(path.join(root, "lib", group), {recursive: true});
    link[group] = [];
    for (let index = 0; index < count; index++) {
      const basename = group === "hiprtc" ? "hiprtc-host-intel.lib" : group === "comgr" ? "amd_comgr.lib" : group + "-" + String(index).padStart(3, "0") + ".lib",
        relative = "lib/" + group + "/" + String(index).padStart(3, "0") + "-" + basename,
        file = path.join(root, relative);
      fs.writeFileSync(file, "!<arch>\n");
      link[group].push(relative); files[relative] = {sha256: fileHash(file), bytes: fs.statSync(file).size};
    }
  }
  const hiprtcMembers = [...hiprtcSourceFiles.map(name => path.basename(name, ".cpp") + ".obj"), "hipRTC_header.obj"],
    comgrArchiveSha256 = files[link["comgr"]?.[0] || ""]?.sha256 || "e".repeat(64),
    resourceReceiptSha256 = "2".repeat(64);
  fs.mkdirSync(path.join(root, "runtime"));
  const runtimeInputs = /** @type {Record<string, {path: string, bytes: number, sha256: string}>} */ ({});
  for (const [name, basename] of Object.entries({cudartStatic: "cudart_static.lib", amdhip64: "amdhip64.lib"})) {
    const relative = "runtime/" + basename, file = path.join(root, relative);
    fs.writeFileSync(file, "!<arch>\n");
    const proof = {path: relative, bytes: fs.statSync(file).size, sha256: fileHash(file)};
    runtimeInputs[name] = proof; files[relative] = {sha256: proof.sha256, bytes: proof.bytes};
  }
  const comgrReceiptSha256 = "d".repeat(64),
    manifest = {version: 3, kind: "windows-native-host-sdk", pins, link, runtimeInputs, files,
      provenance: {llvm: {generationId, receiptSha256: "a".repeat(64), admissionSha256: "b".repeat(64), objectProofSha256: "c".repeat(64)},
        comgr: {receiptSha256: comgrReceiptSha256, archiveSha256: comgrArchiveSha256, llvmGenerationId: generationId},
        acpp: {receiptSha256: "f".repeat(64), sourceRevision: acppRevision, llvmGenerationId: generationId, rocmResourceReceiptSha256: resourceReceiptSha256},
        hiprtc: {receiptSha256: "1".repeat(64), archiveSha256: files[link.hiprtc[0] || ""]?.sha256 || "e".repeat(64), sourceRevision: clrRevision,
          comgrRevision, comgrReceiptSha256, comgrArchiveSha256, resourceReceiptSha256, archiveMembers: hiprtcMembers}},
      buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(root, "component.json"), JSON.stringify(manifest, null, 2) + "\n");
  return manifest;
}

test.describe("Windows static host-component CI", {skip: process.platform !== "linux" && "Linux-hosted Windows build workflow harness"}, () => {
  test("one bounded job wires the 16 external pins and only the final component artifact", () => {
    assert.match(workflow, /runs-on: ubuntu-24\.04/);
    assert.match(workflow, /timeout-minutes: 60\n/);
    assert.match(step("Build and export the static Windows host component"), /timeout-minutes: 45/);
    const jobsSection = workflow.split(/^jobs:\s*$/m)[1] || "";
    assert.equal((jobsSection.match(/^ {2}[A-Za-z0-9_-]+:\s*$/gm) || []).length, 1,
      "the reusable workflow must not grow duplicate intermediate jobs");
    assert.match(workflow, /linux_sdk_bundle_sha256:\n\s+description: Full same-run audited Linux LLVM SDK and source tar, not thin tools/);
    assert.match(workflow, /MOM_NATIVE_LINUX_SDK_BUNDLE_SHA256: \$\{\{ inputs\.linux_sdk_bundle_sha256 \}\}/);
    assert.match(workflow, /MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256: \$\{\{ inputs\.linux_tools_bundle_sha256 \}\}/);
    assert.match(workflow, /build-native-llvm21-sdk"\)\.verifyLinuxLlvmAdmission\(paths\.output\)/);
    assert.match(workflow, /pinnedManifest\(windowsManifestFile, process\.env\[inputPins\.WINDOWS_STATE_MANIFEST\]\)[\s\S]*?winStateManifest = JSON\.parse\(winStateManifestBytes/);
    assert.match(workflow, /native-windows-host-sdk-inputs|composeWindowsLlvmSdkDescriptor/);
    assert.match(workflow, /build-native-rocm-resources\.js/);
    assert.match(workflow, /resourceConfig = ci\.rocmResourcesConfig\(\{source: rocm\.source, llvmSdk: linuxSdk\.sdk\.path, hipIncludeRoot: hipHeaders\.include\}/);
    assert.match(workflow, /cxx: "\/opt\/intel\/oneapi\/compiler\/2026\.2\/bin\/icpx"\}, "win32"\)/);
    assert.match(workflow, /build-native-windows-comgr-sdk\.js/);
    assert.match(workflow, /build-native-windows-hiprtc-sdk\.js/);
    assert.match(workflow, /build-native-windows-acpp-sdk\.js/);
    assert.match(workflow, /exportWindowsHostComponent\(/);
    assert.match(workflow, /Upload only the static Windows host component/);
    assert.match(workflow, /archive: false/);
    const build = runBody(step("Build and export the static Windows host component")),
      resources = build.indexOf('run("scripts/build-native-rocm-resources.js"'),
      comgr = build.indexOf('run("scripts/build-native-windows-comgr-sdk.js", ["build"'),
      hiprtc = build.indexOf('run("scripts/build-native-windows-hiprtc-sdk.js"'),
      acpp = build.indexOf('run("scripts/build-native-windows-acpp-sdk.js"'),
      exportAt = build.indexOf("exportWindowsHostComponent({");
    assert.ok(resources >= 0 && resources < comgr && comgr < hiprtc && hiprtc < acpp && acpp < exportAt,
      "HIPRTC must build after its COMGR/device inputs and before host-component export");
    assert.match(build, /hiprtcSource = path\.join\(rocm\.source, "roots\/clr"\)/);
    assert.match(build, /clone", "--shared", "--no-checkout"/);
    assert.match(build, /checkout", "--detach", hiprtcBuilder\.clrRevision/);
    assert.match(build, /const hiprtcConfig = \{source: hiprtcCheckout, descriptorFile: windowsState\.descriptor, hipInclude: hipHeaders\.include/);
    assert.match(build, /deviceReceipt: resourceResult,[\s\S]*?deviceInstall: path\.join\(resourcesRoot, "install"\), llvmAr: path\.join\(toolInputs\.toolRoot, "llvm-ar"\), output: hiprtcRoot/);
    assert.match(build, /run\("scripts\/build-native-windows-hiprtc-sdk\.js", \[hiprtcConfigFile\], 5 \* 60 \* 1000\)/);
    assert.match(build, /acppResult: path\.join\(acppRoot, "result\.json"\), hiprtcResult, output: componentRoot/);
    const jobHeader = workflow.slice(workflow.indexOf("jobs:"), workflow.indexOf("    steps:"));
    assert.doesNotMatch(jobHeader, /\$\{\{\s*runner\.temp\s*\}\}/, "runner.temp is unavailable in job-level env");
    assert.match(runBody(step("Set the immutable run and 55-minute job deadlines")),
      /MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR=\$RUNNER_TEMP\/mom-native-windows-llvm21-descriptor-\$GITHUB_RUN_ID-\$GITHUB_RUN_ATTEMPT/);
  });

  test("the parent caller maps each same-run external output, including the full Linux SDK pin", () => {
    const start = caller.indexOf("  build-windows-host-component:\n");
    assert.notEqual(start, -1);
    const nextJob = caller.slice(start + 1).search(/^ {2}[A-Za-z0-9_-]+:\s*$/m),
      block = caller.slice(start, nextJob === -1 ? undefined : start + 1 + nextJob);
    assert.match(block, /uses: \.\/\.github\/workflows\/build-windows-host-job\.yml/);
    const mappings = [
      ["windows_llvm_run_deadline_epoch", "windows-llvm-audit", "run_deadline_epoch"],
      ["windows_llvm_state_bundle_sha256", "windows-llvm-audit", "bundle_sha256"],
      ["windows_llvm_state_manifest_sha256", "windows-llvm-audit", "manifest_sha256"],
      ["windows_llvm_admission_sha256", "windows-llvm-audit", "admission_sha256"],
      ["windows_hermes_bundle_sha256", "build-windows-controller-sdk", "hermes_bundle_sha256"],
      ["windows_hermes_handoff_sha256", "build-windows-controller-sdk", "hermes_handoff_sha256"],
      ["linux_tools_bundle_sha256", "install-audit-linux-sdk", "tools_bundle_sha256"],
      ["linux_tools_handoff_sha256", "install-audit-linux-sdk", "tools_handoff_sha256"],
      ["linux_sdk_bundle_sha256", "install-audit-linux-sdk", "bundle_sha256"],
      ["linux_sources_bundle_sha256", "linux-sources", "bundle_sha256"],
      ["dpcpp_cuda_bundle_sha256", "provision-linux-dpcpp-cuda-inputs", "bundle_sha256"],
      ["dpcpp_cuda_manifest_sha256", "provision-linux-dpcpp-cuda-inputs", "manifest_sha256"],
      ["rocm_sources_bundle_sha256", "fetch-linux-rocm-sources", "bundle_sha256"],
      ["acpp_sources_bundle_sha256", "fetch-linux-acpp-sources", "bundle_sha256"],
      ["hip_headers_bundle_sha256", "build-linux-rocm-comgr-sdk", "hip_headers_bundle_sha256"],
      ["hip_headers_handoff_sha256", "build-linux-rocm-comgr-sdk", "hip_headers_handoff_sha256"],
    ];
    for (const [input, job, output] of mappings) {
      assert.match(block, new RegExp("^      " + input + ": \\$\\{\\{ needs\\." + job + "\\.outputs\\." + output + " \\}\\}$", "m"), input);
    }
    assert.match(block, /linux_sdk_bundle_sha256: \$\{\{ needs\.install-audit-linux-sdk\.outputs\.bundle_sha256 \}\}/,
      "the full SDK must not be substituted with the thin compiler-tools archive");
  });

  test("the verified bare CLR input yields the clean ordinary checkout required by HIPRTC admission", t => {
    const directory = tempDir(t), source = path.join(directory, "source"), bare = path.join(directory, "clr.git"),
      checkout = path.join(directory, "checkout"), sourceEnv = {GIT_ALLOW_PROTOCOL: "file", GIT_ATTR_NOSYSTEM: "1",
        GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1",
        GIT_OPTIONAL_LOCKS: "0", GIT_TERMINAL_PROMPT: "0"};
    fs.mkdirSync(source);
    runGit(["init", "--quiet", source]);
    runGit(["-C", source, "config", "user.name", "CI fixture"]);
    runGit(["-C", source, "config", "user.email", "ci@example.invalid"]);
    for (const root of ["hipamd", "rocclr"]) {
      fs.mkdirSync(path.join(source, root)); fs.writeFileSync(path.join(source, root, "source.txt"), root + "\n");
    }
    runGit(["-C", source, "add", "--", "hipamd", "rocclr"]);
    runGit(["-C", source, "commit", "--quiet", "-m", "fixture"]);
    runGit(["clone", "--quiet", "--bare", "--", source, bare]);
    const revision = runGit(["-C", bare, "rev-parse", "HEAD"]);
    assert.equal(runGit(["-C", bare, "rev-parse", "--is-bare-repository"]), "true");
    runGit(["clone", "--shared", "--no-checkout", "--", bare, checkout], sourceEnv);
    runGit(["-C", checkout, "checkout", "--detach", revision], sourceEnv);
    assert.equal(runGit(["-C", checkout, "rev-parse", "HEAD"]), revision);
    assert.equal(runGit(["-C", checkout, "rev-parse", "--is-inside-work-tree"]), "true");
    assert.equal(runGit(["-C", checkout, "status", "--porcelain=v1", "--untracked-files=all", "--", "hipamd", "rocclr"]), "");
  });

  test("the test step installs tools before tests and every named test file exists", () => {
    const extraction = step("Install SDK extraction and test tools"), tests = step("Test host-component producer and same-run input contracts"),
      command = runCommand(tests).trim(), testNames = /** @type {string[]} */ (command.slice(command.indexOf("--test")).match(/tests\/[^\s]+\.js/g) || []);
    assert.match(extraction, /7zip msitools unzip/);
    assert.ok(workflow.indexOf(extraction) < workflow.indexOf(tests), "system test prerequisites must be installed first");
    assert.ok(testNames.includes("tests/native_windows_cudart.js"));
    assert.ok(testNames.includes("tests/native_windows_amdhip.js"));
    assert.ok(testNames.includes("tests/native_windows_hiprtc_sdk.js"));
    assert.ok(testNames.includes("tests/native_windows_host_ci_job.js"));
    assert.equal(testNames.length, 15);
    for (const name of testNames) {assert.ok(fs.statSync(path.join(repo, name)).isFile(), "missing named test: " + name);}
  });

  test("Windows cross-build tool installation bounds mirror stalls and preserves APT failures", t => {
    const directory = tempDir(t), log = path.join(directory, "apt.log"),
      llvmWorkflow = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-llvm-job.yml"), "utf8");
    for (const block of [step("Install SDK extraction and test tools"),
      step("Install archive and MSI extraction tools", llvmWorkflow)]) {
      const source = 'sudo() { printf "%s\\n" "$*" >> "$MOM_APT_LOG"; return "$MOM_APT_EXIT"; }\n' +
        "test() { :; }\n" + runBody(block);
      assert.match(block, /timeout-minutes: 8/);
      for (const exit of [0, 42]) {
        fs.writeFileSync(log, "");
        const result = runBash(source, {PATH: "/usr/bin:/bin", MOM_APT_LOG: log, MOM_APT_EXIT: String(exit)}),
          calls = fs.readFileSync(log, "utf8").trim().split("\n");
        assert.equal(result.status, exit, result.stderr);
        assert.equal(calls.length, exit ? 1 : 2, "a failed index refresh must stop before installation");
        for (const command of calls) {
          for (const option of ["Acquire::Retries=2", "Acquire::http::Timeout=20", "Acquire::https::Timeout=20"]) {
            assert.ok(command.includes("-o " + option), command);
          }
          assert.doesNotMatch(command, /allow-unauthenticated|AllowInsecureRepositories/);
        }
      }
    }
  });

  test("the actual deadline shell block caps the job at 55 minutes and rejects a short run budget", t => {
    const directory = tempDir(t), block = runBody(step("Set the immutable run and 55-minute job deadlines")),
      output = path.join(directory, "output"), environment = {PATH: process.env["PATH"] || "/usr/bin:/bin", RUNNER_TEMP: directory,
        GITHUB_ENV: path.join(directory, "env"), GITHUB_OUTPUT: output, GITHUB_RUN_ID: "123456", GITHUB_RUN_ATTEMPT: "2"},
      now = Math.floor(Date.now() / 1000), deadline = String(now + 3600);
    fs.writeFileSync(environment.GITHUB_ENV, ""); fs.writeFileSync(output, "");
    const good = runBash(block, {...environment, REQUESTED_RUN_DEADLINE: deadline});
    assert.equal(good.status, 0, good.stderr);
    const jobDeadline = Number(fs.readFileSync(environment.GITHUB_ENV, "utf8").match(/^MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH=(\d+)$/m)?.[1]);
    const checkedAt = Math.floor(Date.now() / 1000);
    assert.ok(jobDeadline > 0 && jobDeadline <= checkedAt + 3300 && jobDeadline > checkedAt + 3200);
    const prior = fs.readFileSync(output, "utf8"), priorEnv = fs.readFileSync(environment.GITHUB_ENV, "utf8");
    const short = runBash(block, {...environment, REQUESTED_RUN_DEADLINE: String(now + 120)});
    assert.notEqual(short.status, 0);
    assert.match(short.stderr, /Insufficient original run budget/);
    assert.equal(fs.readFileSync(output, "utf8"), prior, "rejected deadline must not publish output values");
    assert.equal(fs.readFileSync(environment.GITHUB_ENV, "utf8"), priorEnv, "rejected deadline must not mutate the job environment");
  });

  test("the actual pin shell block rejects malformed pins before writing artifact names", t => {
    const directory = tempDir(t), block = runBody(step("Validate external pins and derive exact same-run artifact names")),
      names = ["WINDOWS_STATE_BUNDLE", "WINDOWS_STATE_MANIFEST", "WINDOWS_ADMISSION", "HERMES_BUNDLE", "HERMES_HANDOFF",
        "LINUX_TOOLS_BUNDLE", "LINUX_TOOLS_HANDOFF", "LINUX_SDK_BUNDLE", "LINUX_SOURCES_BUNDLE", "DPCPP_CUDA_BUNDLE",
        "DPCPP_CUDA_MANIFEST", "ROCM_SOURCES_BUNDLE", "ACPP_SOURCES_BUNDLE", "HIP_HEADERS_BUNDLE", "HIP_HEADERS_HANDOFF"],
      environment = {PATH: process.env["PATH"] || "/usr/bin:/bin", GITHUB_OUTPUT: path.join(directory, "output"), GITHUB_RUN_ID: "123456", GITHUB_RUN_ATTEMPT: "2",
        ...Object.fromEntries(names.map(name => [name, "a".repeat(64)]))};
    fs.writeFileSync(environment.GITHUB_OUTPUT, "");
    const good = runBash(block, environment);
    assert.equal(good.status, 0, good.stderr);
    const output = fs.readFileSync(environment.GITHUB_OUTPUT, "utf8");
    assert.match(output, /^linux_sdk_archive=native-linux-llvm21-sdk-123456-2\.tar$/m);
    assert.match(output, /^windows_state_manifest=native-windows-llvm-state-123456-2-audited\.json$/m);
    const bad = runBash(block, {...environment, DPCPP_CUDA_MANIFEST: "f".repeat(63)});
    assert.notEqual(bad.status, 0);
    assert.match(bad.stderr, /Missing or malformed external artifact pin/);
    assert.equal(fs.readFileSync(environment.GITHUB_OUTPUT, "utf8"), output, "bad pins must not emit any artifact names");
  });

  test("the real inline builder parses and fails closed on a malformed external pin before work", t => {
    const directory = tempDir(t), block = runBody(step("Build and export the static Windows host component")),
      script = block.match(/(?:^|\n)node <<'NODE'\n([\s\S]*?)\nNODE(?:\n|$)/m)?.[1];
    assert.ok(script, "expected the single inline Node host builder");
    const output = path.join(directory, "output"), summary = path.join(directory, "summary");
    fs.writeFileSync(output, "sentinel\n"); fs.writeFileSync(summary, "summary\n");
    const environment = {...process.env, MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256: "not-a-digest", GITHUB_OUTPUT: output,
        GITHUB_STEP_SUMMARY: summary, GITHUB_RUN_ID: "123456", GITHUB_RUN_ATTEMPT: "2"},
      result = spawnSync(process.execPath, ["-e", script], {cwd: repo, env: environment, encoding: "utf8", timeout: 10_000});
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /malformed WINDOWS_STATE_BUNDLE pin/);
    assert.equal(fs.readFileSync(output, "utf8"), "sentinel\n");
    assert.equal(fs.readFileSync(summary, "utf8"), "summary\n");
    assert.match(script, /restoreWindowsNativeLlvmState\(/);
    assert.match(script, /auditDpcppCudaInputs\(/);
    assert.match(script, /verifyRocmSources\(/);
    assert.match(script, /verifyAcppSource\(/);
    assert.match(script, /rocmHipHeaderInputs\(/);
    assert.match(script, /prepareWindowsCudart/);
    assert.match(script, /windowsAmdhipInputs\(/);
    assert.match(script, /verifyLinuxLlvmAdmission\(/);
    assert.match(script, /verifyWindowsLlvmAdmission\(/);
    assert.match(script, /build-native-rocm-resources\.js/);
    assert.match(script, /build-native-windows-comgr-sdk\.js/);
    assert.match(script, /build-native-windows-acpp-sdk\.js/);
    assert.match(script, /exportWindowsHostComponent\(/);
  });
  test("HIP headers retain their exported transport root through host admission and ACPP", () => {
    const script = runBody(step("Build and export the static Windows host component"));
    assert.match(script, /hipName = "native-rocm-hip-headers-" \+ suffix/);
    assert.match(script, /restore\(hipArchive, "HIP_HEADERS_BUNDLE", \[hipName\], "\/tmp"/);
    assert.match(script, /rocmHipHeaderInputs\(path\.join\("\/tmp", hipName\)/);
    assert.match(script, /rocmRoot = hipHeaders\.root/);
  });
  test("Linux LLVM tools restore beneath the original configure-time absolute root", () => {
    const script = runBody(step("Build and export the static Windows host component"));
    assert.match(script, /toolsRoot = path\.join\(paths\.runnerTemp, `linux-llvm-tools-extracted-\$\{suffix\}`\)/);
    assert.match(script, /toolsBundle = path\.join\(toolsRoot, toolName\)/);
    assert.match(script, /restore\(toolsArchive, "LINUX_TOOLS_BUNDLE", \[toolName\], toolsRoot, 1024 \*\* 3, 25 \* 60 \* 1000, true\)/);
    assert.match(script, /linuxLlvmToolHandoffInputs\(toolsBundle,/);
    assert.match(script, /verifyWindowsLlvmAdmission\(windowsState\.output/);
  });
  test("the actual restore helper unpacks a raw archive only after external pin admission", t => {
    const runId = String(Date.now()) + String(process.pid).padStart(6, "0"), attempt = "2", suffix = runId + "-" + attempt,
      runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-host-restore-test-")), output = path.join(runnerTemp, "output"),
      summary = path.join(runnerTemp, "summary"), block = runBody(step("Build and export the static Windows host component")),
      script = block.match(/(?:^|\n)node <<'NODE'\n([\s\S]*?)\nNODE(?:\n|$)/m)?.[1],
      terminal = 'main().catch(error => {console.error(error instanceof Error ? error.stack || error.message : "Windows host component failed"); process.exitCode = 1;});';
    assert.ok(script, "expected the inline same-run input restore helper");
    const hipRestore = script.match(/^\s*restore\(hipArchive,.*$/m)?.[0].trim(),
      toolRestore = script.match(/^\s*restore\(toolsArchive,.*$/m)?.[0].trim();
    assert.ok(hipRestore, "exercise the actual HIP archive restore call, not a synthetic root selection");
    assert.ok(toolRestore, "exercise the same-run Linux LLVM tool restore call");
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    t.after(() => fs.rmSync(path.join("/tmp", "native-rocm-hip-headers-" + suffix), {recursive: true, force: true}));
    const rootVars = {
        MOM_NATIVE_SOURCE_DIR: "/tmp/mom-native-prerequisites-" + suffix,
        MOM_NATIVE_LINUX_SDK_DIR: "/tmp/mom-native-llvm21-sdk-" + suffix,
        MOM_NATIVE_IO_SOURCE_DIR: "/tmp/mom-native-io-prerequisites-" + suffix,
        MOM_NATIVE_IO_SDK_DIR: "/tmp/mom-native-io-openssl-sdk-" + suffix,
        MOM_NATIVE_LINUX_SOURCES_DIR: "/tmp/mom-native-linux-source-prerequisites-" + suffix,
        MOM_NATIVE_ZSTD_SOURCE_DIR: "/tmp/mom-native-zstd-prerequisites-" + suffix,
        MOM_NATIVE_ZSTD_SDK_DIR: "/tmp/mom-native-zstd-sdk-" + suffix,
        MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: "/tmp/mom-native-linux-jit-resources-" + suffix,
        MOM_NATIVE_ROCM_COMGR_SDK_DIR: "/tmp/mom-native-rocm-comgr-sdk-" + suffix,
        MOM_NATIVE_ROCM_RESOURCES_DIR: "/tmp/mom-native-rocm-resources-" + suffix,
        MOM_NATIVE_ROCM_HIP_HEADERS_DIR: "/tmp/mom-native-rocm-hip-headers-" + suffix,
        MOM_NATIVE_ROCM_HOST_SDK_DIR: "/tmp/mom-native-rocm-host-sdk-" + suffix,
        MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: "/tmp/mom-native-linux-sdk-descriptor-" + suffix,
      },
      pinVars = ["MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256",
        "MOM_NATIVE_WINDOWS_LLVM_ADMISSION_SHA256", "MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256",
        "MOM_NATIVE_WINDOWS_LLVM_HERMES_HANDOFF_SHA256", "MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256",
        "MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256", "MOM_NATIVE_LINUX_SDK_BUNDLE_SHA256", "MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256",
        "MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256", "MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256",
        "MOM_NATIVE_ROCM_SOURCE_BUNDLE_SHA256", "MOM_NATIVE_ACPP_SOURCE_BUNDLE_SHA256",
        "MOM_NATIVE_ROCM_HIP_HEADERS_BUNDLE_SHA256", "MOM_NATIVE_ROCM_HIP_HEADERS_HANDOFF_SHA256"],
      env = {...process.env, ...rootVars, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", RUNNER_TEMP: runnerTemp,
        GITHUB_RUN_ID: runId, GITHUB_RUN_ATTEMPT: attempt, GITHUB_JOB: "windows-host-component",
        MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(Math.floor(Date.now() / 1000) + 600),
        MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.floor(Date.now() / 1000) + 600),
        GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary,
        ...Object.fromEntries(pinVars.map(name => [name, "a".repeat(64)]))};
    fs.writeFileSync(output, ""); fs.writeFileSync(summary, "");
    const testTail = [
      "const manifestFile = path.join(paths.runnerTemp, 'fixture-manifest.json'), manifestBytes = Buffer.from(JSON.stringify({fixture: true}) + '\\n');",
      "fs.writeFileSync(manifestFile, manifestBytes, {flag: 'wx'});",
      "const manifestPin = fileHash(manifestFile), admittedManifest = pinnedManifest(manifestFile, manifestPin, 1024);",
      "if (!admittedManifest.bytes.equals(manifestBytes) || admittedManifest.sha256 !== manifestPin || JSON.parse(admittedManifest.bytes.toString('utf8')).fixture !== true) {",
      "  throw new Error('actual manifest helper did not hash and return bounded pinned bytes');",
      "}",
      "for (const [pin, maximum] of [['f'.repeat(64), 1024], [manifestPin, 1]]) {",
      "  let rejected = false; try {pinnedManifest(manifestFile, pin, maximum);} catch {rejected = true;}",
      "  if (!rejected) {throw new Error('actual manifest helper accepted a bad pin or byte bound');}",
      "}",
      "const rootName = 'native-linux-llvm-tools-' + suffix, sourceBase = path.join(paths.runnerTemp, 'fixture-input'),",
      "  sourceRoot = path.join(sourceBase, rootName), archive = path.join(paths.runnerTemp, rootName + '.tar'),",
      "  toolsRoot = path.join(paths.runnerTemp, 'linux-llvm-tools-extracted-' + suffix), clangTool = path.join(sourceRoot, 'bin', 'clang-tblgen');",
      "fs.mkdirSync(path.dirname(clangTool), {recursive: true}); fs.writeFileSync(clangTool, 'fixture clang-tblgen\\n');",
      "ci.packState(archive, sourceBase, [rootName]);",
      "process.env[inputPins.LINUX_TOOLS_BUNDLE] = 'f'.repeat(64);",
      "let rejectedWrongToolsPin = false; try {restore(archive, 'LINUX_TOOLS_BUNDLE', [rootName], toolsRoot, 1024, 25 * 60 * 1000, true);} catch {rejectedWrongToolsPin = true;}",
      "if (!rejectedWrongToolsPin || fs.existsSync(toolsRoot)) {throw new Error('wrong Linux tools archive pin created its extraction root');}",
      "process.env[inputPins.LINUX_TOOLS_BUNDLE] = fileHash(archive);",
      "const wrongToolsBase = path.join(paths.runnerTemp, 'alternate-linux-llvm-tools-root');",
      "let rejectedWrongToolsBase = false; try {restore(archive, 'LINUX_TOOLS_BUNDLE', [rootName], wrongToolsBase, 1024, 25 * 60 * 1000, true);} catch {rejectedWrongToolsBase = true;}",
      "if (!rejectedWrongToolsBase || fs.existsSync(wrongToolsBase)) {throw new Error('wrong Linux tools extraction parent was created');}",
      "const wrongRootName = 'alternate-linux-llvm-tools-' + suffix, wrongRootSource = path.join(sourceBase, wrongRootName),",
      "  wrongRootArchive = path.join(paths.runnerTemp, wrongRootName + '.tar');",
      "fs.mkdirSync(path.join(wrongRootSource, 'bin'), {recursive: true}); fs.writeFileSync(path.join(wrongRootSource, 'bin', 'clang-tblgen'), 'wrong root\\n');",
      "ci.packState(wrongRootArchive, sourceBase, [wrongRootName]);",
      "process.env[inputPins.LINUX_TOOLS_BUNDLE] = fileHash(wrongRootArchive);",
      "let rejectedWrongToolsRoot = false; try {restore(wrongRootArchive, 'LINUX_TOOLS_BUNDLE', [wrongRootName], toolsRoot, 1024, 25 * 60 * 1000, true);} catch {rejectedWrongToolsRoot = true;}",
      "if (!rejectedWrongToolsRoot || fs.existsSync(toolsRoot)) {throw new Error('wrong Linux tools archive root created the extraction directory');}",
      "process.env[inputPins.LINUX_TOOLS_BUNDLE] = fileHash(archive);",
      "fs.mkdirSync(toolsRoot, {mode: 0o700}); const existingRootSentinel = path.join(toolsRoot, 'sentinel'); fs.writeFileSync(existingRootSentinel, 'preserve\\n');",
      "let rejectedExistingToolsRoot = false; try {restore(archive, 'LINUX_TOOLS_BUNDLE', [rootName], toolsRoot, 1024, 25 * 60 * 1000, true);} catch {rejectedExistingToolsRoot = true;}",
      "if (!rejectedExistingToolsRoot || fs.readFileSync(existingRootSentinel, 'utf8') !== 'preserve\\n' || fs.existsSync(path.join(toolsRoot, rootName))) {",
      "  throw new Error('pre-existing Linux tools extraction root was overwritten');",
      "}",
      "fs.rmSync(toolsRoot, {recursive: true, force: true});",
      "const restored = restore(archive, 'LINUX_TOOLS_BUNDLE', [rootName], toolsRoot, 1024, 25 * 60 * 1000, true),",
      "  restoredClangTool = path.join(toolsRoot, rootName, 'bin', 'clang-tblgen');",
      "if (restored.expandedBytes !== Buffer.byteLength('fixture clang-tblgen\\n') ||",
      "    fs.readFileSync(restoredClangTool, 'utf8') !== 'fixture clang-tblgen\\n') {",
      "  throw new Error('same-run Linux LLVM tools were not restored at the configure-time clang-tblgen path');",
      "}",
      "const hipName = 'native-rocm-hip-headers-' + suffix, hipArchive = path.join(paths.runnerTemp, hipName + '.tar');",
      "fs.mkdirSync(path.join(sourceBase, hipName)); fs.writeFileSync(path.join(sourceBase, hipName, 'marker'), 'headers');",
      "ci.packState(hipArchive, sourceBase, [hipName]); process.env[inputPins.HIP_HEADERS_BUNDLE] = fileHash(hipArchive);",
      hipRestore,
      "if (fs.readFileSync(path.join('/tmp', hipName, 'marker'), 'utf8') !== 'headers') {throw new Error('HIP transport root was not restored');}",
    ].join("\n");
    assert.ok(script.includes(terminal), "inline builder must have its normal main terminal");
    const program = script.replace(terminal, testTail), result = spawnSync(process.execPath, ["-e", program],
      {cwd: repo, env, encoding: "utf8", timeout: 30_000});
    assert.equal(result.status, 0, result.stderr);
    assert.match(program, /pinnedManifest\(windowsManifestFile,[\s\S]*?JSON\.parse\(winStateManifestBytes/);
    assert.match(program, /pinnedArchive\(file, pinName\);[\s\S]*?ci\.extractState\(file, base/);
  });
  test("the real seal heredoc re-admits, packs and rechecks an exact host-component fixture", t => {
    const runId = String(Date.now()) + String(process.pid).padStart(6, "0"), attempt = "2", suffix = runId + "-" + attempt,
      root = path.join("/tmp", "native-windows-host-component-" + suffix), runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-host-seal-test-")),
      output = path.join(runnerTemp, "output"), summary = path.join(runnerTemp, "summary"), name = path.basename(root),
      block = runBody(step("Seal the archive-only component artifact")), script = block.match(/(?:^|\n)node <<'NODE'\n([\s\S]*?)\nNODE(?:\n|$)/m)?.[1];
    assert.ok(script, "expected the inline public component archive seal");
    assert.equal(fs.existsSync(root), false, "fixture root must be fresh");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    hostComponentFixture(root);
    fs.writeFileSync(output, ""); fs.writeFileSync(summary, "");
    const env = {...process.env, RUNNER_TEMP: runnerTemp, GITHUB_RUN_ID: runId, GITHUB_RUN_ATTEMPT: attempt,
        MOM_NATIVE_WINDOWS_HOST_COMPONENT_ROOT: root,
        MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.floor(Date.now() / 1000) + 600),
        GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary},
      result = spawnSync(process.execPath, ["-e", script], {cwd: repo, env, encoding: "utf8", timeout: 30_000}),
      archive = path.join(runnerTemp, name + ".tar");
    assert.equal(result.status, 0, result.stderr);
    const outputText = fs.readFileSync(output, "utf8"), rows = Object.fromEntries(outputText.trim().split(/\r?\n/).map(line => line.split("=")));
    assert.equal(rows.artifact_name, name + ".tar");
    assert.match(rows.bundle_sha256 || "", /^[a-f0-9]{64}$/);
    assert.match(rows.manifest_sha256 || "", /^[a-f0-9]{64}$/);
    const ci = require("../scripts/prepare-native-ci"), {fileHash} = require("../scripts/build-source-inputs"),
      members = ci.listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort();
    assert.equal(fileHash(archive), rows.bundle_sha256);
    assert.ok(members.includes(name + "/component.json"));
    assert.equal(ci.tarExpandedBytes(archive, members.length) > 0, true);
    assert.match(fs.readFileSync(summary, "utf8"), /No runtime execution, GPU validation or redistribution qualification/);
  });
});
