"use strict";

const fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs");
const {windowsGpuCiJobPlan, windowsGpuCiJobPreflight, windowsGpuHostSdkPins, windowsGpuDpcppManifest,
  admitWindowsGpuDpcppInputs, windowsGpuLibraryPath, windowsGpuAcppConfigInput} =
  require("../scripts/run-native-windows-gpu-ci");
const repo = path.resolve(__dirname, "..");

let runCounter = 0;
const managedKeys = ["RUNNER_TEMP", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY",
  "MOM_AOT_BUILD_ONLY", "MOM_BUILD_JOBS", "MOM_NATIVE_SOURCE_DIR", "MOM_NATIVE_LINUX_SDK_DIR", "MOM_NATIVE_IO_SOURCE_DIR",
  "MOM_NATIVE_IO_SDK_DIR", "MOM_NATIVE_LINUX_SOURCES_DIR", "MOM_NATIVE_ZSTD_SOURCE_DIR", "MOM_NATIVE_ZSTD_SDK_DIR",
  "MOM_NATIVE_LINUX_JIT_RESOURCES_DIR", "MOM_NATIVE_ROCM_COMGR_SDK_DIR", "MOM_NATIVE_ROCM_RESOURCES_DIR",
  "MOM_NATIVE_ROCM_HIP_HEADERS_DIR", "MOM_NATIVE_ROCM_HOST_SDK_DIR", "MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR",
  "MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR", "MOM_NATIVE_WINDOWS_GPU_DIR", "MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH",
  "MOM_NATIVE_WINDOWS_GPU_SOURCE_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_GPU_DPCPP_BUNDLE_SHA256",
  "MOM_NATIVE_WINDOWS_GPU_DPCPP_MANIFEST_SHA256", "MOM_NATIVE_WINDOWS_GPU_TOOLS_BUNDLE_SHA256",
  "MOM_NATIVE_WINDOWS_GPU_TOOLS_HANDOFF_SHA256", "MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256",
  "MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256",
  "MOM_NATIVE_ACPP_SOURCE_DIR", "MOM_NATIVE_WINDOWS_GPU_ACPP_SOURCE_BUNDLE_SHA256",
  "MOM_NATIVE_WINDOWS_GPU_ACPP_SDK_BUNDLE_SHA256", "MOM_NATIVE_WINDOWS_GPU_FRONTEND_BUNDLE_SHA256",
  "MOM_NATIVE_WINDOWS_GPU_FRONTEND_HANDOFF_SHA256"];

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const previous = new Map(managedKeys.map(key => [key, process.env[key]])), runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-gpu-ci-")),
    run = String(Date.now()) + String(++runCounter).padStart(4, "0"), attempt = "1", suffix = run + "-" + attempt,
    incoming = path.join(runnerTemp, "incoming"), output = path.join(runnerTemp, "github-output"), summary = path.join(runnerTemp, "github-summary"),
    tempRoot = "/tmp", dpcppName = "mom-native-dpcpp-cuda-inputs-" + suffix,
    files = [path.join(incoming, "native-linux-sdk-state-" + suffix + "-sources.tar"),
      path.join(incoming, "native-linux-dpcpp-cuda-inputs-" + suffix + ".tar"), path.join(incoming, "native-linux-llvm-tools-" + suffix + ".tar")];
  fs.mkdirSync(incoming); fs.writeFileSync(output, "", {flag: "wx"}); fs.writeFileSync(summary, "", {flag: "wx"});
  /** @param {string} name */
  const suffixPath = name => path.join(tempRoot, name + "-" + suffix);
  const environment = /** @type {NodeJS.ProcessEnv} */ ({
    RUNNER_TEMP: runnerTemp, GITHUB_RUN_ID: run, GITHUB_RUN_ATTEMPT: attempt, GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary,
    MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
    MOM_NATIVE_SOURCE_DIR: suffixPath("mom-native-prerequisites"), MOM_NATIVE_LINUX_SDK_DIR: suffixPath("mom-native-llvm21-sdk"),
    MOM_NATIVE_IO_SOURCE_DIR: suffixPath("mom-native-io-prerequisites"), MOM_NATIVE_IO_SDK_DIR: suffixPath("mom-native-io-openssl-sdk"),
    MOM_NATIVE_LINUX_SOURCES_DIR: suffixPath("mom-native-linux-source-prerequisites"),
    MOM_NATIVE_ZSTD_SOURCE_DIR: suffixPath("mom-native-zstd-prerequisites"), MOM_NATIVE_ZSTD_SDK_DIR: suffixPath("mom-native-zstd-sdk"),
    MOM_NATIVE_LINUX_JIT_RESOURCES_DIR: suffixPath("mom-native-linux-jit-resources"),
    MOM_NATIVE_ROCM_COMGR_SDK_DIR: suffixPath("mom-native-rocm-comgr-sdk"),
    MOM_NATIVE_ROCM_RESOURCES_DIR: suffixPath("mom-native-rocm-resources"),
    MOM_NATIVE_ROCM_HIP_HEADERS_DIR: suffixPath("mom-native-rocm-hip-headers"),
    MOM_NATIVE_ROCM_HOST_SDK_DIR: suffixPath("mom-native-rocm-host-sdk"),
    MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR: suffixPath("mom-native-linux-sdk-descriptor"),
    MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR: path.join(tempRoot, dpcppName),
    MOM_NATIVE_WINDOWS_GPU_DIR: suffixPath("mom-native-windows-gpu-oneapi"),
    MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 40 * 60 * 1000) / 1000)),
    MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256: "",
    MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_BUNDLE_SHA256: "", MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256: "",
  });
  for (const name of ["SOURCE_BUNDLE", "DPCPP_BUNDLE", "DPCPP_MANIFEST", "TOOLS_BUNDLE", "TOOLS_HANDOFF"]) {
    environment["MOM_NATIVE_WINDOWS_GPU_" + name + "_SHA256"] = "0".repeat(64);
  }
  for (const [key, value] of Object.entries(environment)) {process.env[key] = value;}
  for (const [index, file] of files.entries()) {fs.writeFileSync(file, Buffer.from("metadata preflight fixture " + index), {flag: "wx"});}
  process.env["MOM_NATIVE_WINDOWS_GPU_SOURCE_BUNDLE_SHA256"] = fileHash(files[0] || "");
  process.env["MOM_NATIVE_WINDOWS_GPU_DPCPP_BUNDLE_SHA256"] = fileHash(files[1] || "");
  process.env["MOM_NATIVE_WINDOWS_GPU_TOOLS_BUNDLE_SHA256"] = fileHash(files[2] || "");
  t.after(() => {
    for (const key of managedKeys) {
      const value = previous.get(key);
      if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
    }
    fs.rmSync(runnerTemp, {recursive: true, force: true});
  });
  return {runnerTemp, output, summary, files, suffix};
}

/** @param {ReturnType<typeof fixture>} f @param {"acpp-cuda"|"acpp-hip"} profile */
function addAcppArtifacts(f, profile) {
  process.env["MOM_NATIVE_WINDOWS_GPU_DIR"] = path.join("/tmp", "mom-native-windows-gpu-" + profile + "-" + f.suffix);
  process.env["MOM_NATIVE_ACPP_SOURCE_DIR"] = path.join("/tmp", "mom-native-acpp-prerequisites-" + f.suffix);
  const incoming = path.join(f.runnerTemp, "incoming"), source = path.join(incoming, "native-acpp-state-" + f.suffix + "-sources.tar"),
    sdk = path.join(incoming, "native-linux-acpp-cuda-hip-sdk-" + f.suffix + ".tar"),
    frontend = path.join(incoming, "native-linux-llvm-acpp-frontend-" + f.suffix + ".tar");
  const archives = /** @type {Array<[string,string]>} */ ([[source, "ACPP_SOURCE_BUNDLE"], [sdk, "ACPP_SDK_BUNDLE"], [frontend, "FRONTEND_BUNDLE"]]);
  for (const [file, key] of archives) {
    fs.writeFileSync(file, "metadata fixture " + key, {flag: "wx"});
    process.env["MOM_NATIVE_WINDOWS_GPU_" + key + "_SHA256"] = fileHash(file);
  }
  process.env["MOM_NATIVE_WINDOWS_GPU_FRONTEND_HANDOFF_SHA256"] = "f".repeat(64);
  if (profile === "acpp-cuda") {
    const images = path.join(incoming, "native-linux-cuda-images-" + f.suffix + ".tar");
    fs.writeFileSync(images, "metadata fixture CUDA images", {flag: "wx"});
    process.env["MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256"] = fileHash(images);
  } else {
    delete process.env["MOM_NATIVE_WINDOWS_GPU_DPCPP_BUNDLE_SHA256"];
    delete process.env["MOM_NATIVE_WINDOWS_GPU_DPCPP_MANIFEST_SHA256"];
    process.env["MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256"] = "";
    const headers = path.join(incoming, "native-rocm-hip-headers-" + f.suffix + ".tar");
    fs.writeFileSync(headers, "metadata fixture HIP headers", {flag: "wx"});
    process.env["MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_BUNDLE_SHA256"] = fileHash(headers);
    process.env["MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256"] = "e".repeat(64);
  }
}

test.describe("Windows GPU CI job preflight", {skip: process.platform !== "linux" && "Linux Windows GPU factory job"}, () => {
  test("derives exact same-run paths and requires all five external pins", t => {
    const f = fixture(t), plan = windowsGpuCiJobPlan("oneapi");
    assert.equal(plan.suffix, f.suffix);
    assert.equal(plan.sourceTar, f.files[0]);
    assert.equal(plan.dpcppTar, f.files[1]);
    assert.equal(plan.toolsTar, f.files[2]);
    assert.equal(plan.factory.profile, "oneapi");
    assert.equal(plan.factory.root, path.join("/tmp", "mom-native-windows-gpu-oneapi-" + f.suffix));
    assert.equal(plan.configSource, path.join(f.runnerTemp, "windows-gpu-oneapi-config-" + f.suffix + ".json"));
    assert.throws(() => windowsGpuCiJobPlan("unsupported"), /five maintained two-job factories/);
    process.env["MOM_NATIVE_WINDOWS_GPU_TOOLS_HANDOFF_SHA256"] = "not-a-sha";
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /every selected input's external same-run SHA-256 pin/);
  });

  test("admits pinned archive metadata and empty distinct reports without creating producer roots", t => {
    const f = fixture(t), plan = windowsGpuCiJobPreflight("oneapi");
    assert.equal(plan.suffix, f.suffix);
    assert.equal(fs.readFileSync(f.output, "utf8"), "");
    assert.equal(fs.readFileSync(f.summary, "utf8"), "");
    assert.equal(fs.existsSync(plan.factory.root), false);
    assert.equal(fs.existsSync(plan.dpcpp?.root || ""), false);
    assert.equal(fs.existsSync(plan.toolsParent), false);
    assert.match(plan.pins["DPCPP_MANIFEST"] || "", /^[a-f0-9]{64}$/);
    assert.ok(f.files.every(file => fs.existsSync(file)), "preflight verifies but does not retire artifact inputs");
  });

  test("rejects an expired or insufficient shared job deadline before mutation", t => {
    const f = fixture(t), plan = windowsGpuCiJobPlan("oneapi");
    process.env["MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"] = String(Math.floor(Date.now() / 1000) - 1);
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /deadline is expired or exceeds the 60-minute job bound/);
    process.env["MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"] = String(Math.ceil((Date.now() + 12 * 60 * 1000) / 1000));
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /Insufficient aggregate job time/);
    assert.equal(fs.existsSync(plan.factory.root), false);
    assert.ok(f.files.every(file => fs.existsSync(file)));
  });

  test("rejects report aliasing and changed archives before any input retirement", t => {
    const f = fixture(t), plan = windowsGpuCiJobPlan("oneapi"), reportBytes = fs.readFileSync(f.output);
    process.env["GITHUB_STEP_SUMMARY"] = f.output;
    assert.throws(() => windowsGpuCiJobPreflight("oneapi"), /reports alias one another/);
    process.env["GITHUB_STEP_SUMMARY"] = f.summary;
    process.env["GITHUB_OUTPUT"] = plan.sourceTar;
    assert.throws(() => windowsGpuCiJobPreflight("oneapi"), /canonical empty files/);
    process.env["GITHUB_OUTPUT"] = f.output;
    fs.appendFileSync(plan.dpcppTar || "", "tamper");
    assert.throws(() => windowsGpuCiJobPreflight("oneapi"), /differs from its external pin/);
    assert.deepEqual(fs.readFileSync(f.output), reportBytes);
    assert.ok(f.files.every(file => fs.existsSync(file)));
    assert.equal(fs.existsSync(plan.factory.root), false);
    assert.equal(fs.existsSync(plan.dpcpp?.root || ""), false);
  });

  test("rejects a stale factory handoff output before restore", t => {
    const f = fixture(t), plan = windowsGpuCiJobPlan("oneapi");
    fs.writeFileSync(plan.factory.archive, "stale output", {flag: "wx"});
    assert.throws(() => windowsGpuCiJobPreflight("oneapi"), /fresh canonical input\/output roots/);
    assert.ok(f.files.every(file => fs.existsSync(file)), "stale-output rejection precedes restore and input retirement");
    assert.equal(fs.readFileSync(plan.factory.archive, "utf8"), "stale output");
    assert.equal(fs.existsSync(plan.factory.root), false);
  });

  test("rejects a non-two-job environment before deriving factory paths", t => {
    fixture(t); process.env["MOM_BUILD_JOBS"] = "4";
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /five maintained two-job factories/);
  });

  test("binds host-SDK pins to producer receipt fields and rejects repinned metadata drift", t => {
    const f = fixture(t), root = path.join(f.runnerTemp, "host-sdk"); fs.mkdirSync(root);
    const values = {descriptor: {sha256: "a".repeat(64)}, provenance: {sha256: "b".repeat(64)}, overlay: {sha256: "c".repeat(64)}},
      resultFile = path.join(root, "result.json");
    fs.writeFileSync(resultFile, JSON.stringify(values) + "\n", {flag: "wx"});
    const producer = {sdk_receipt_sha256: fileHash(resultFile), descriptor_sha256: values.descriptor.sha256,
      files_provenance_sha256: values.provenance.sha256};
    assert.deepEqual(windowsGpuHostSdkPins(root, producer), {resultSha256: producer.sdk_receipt_sha256,
      descriptorSha256: producer.descriptor_sha256, filesSha256: producer.files_provenance_sha256, overlaySha256: values.overlay.sha256});
    const changed = {...values, descriptor: {sha256: "d".repeat(64)}};
    fs.writeFileSync(resultFile, JSON.stringify(changed) + "\n");
    assert.throws(() => windowsGpuHostSdkPins(root, producer), /receipt changed after its producer/);
    const repinned = {...producer, sdk_receipt_sha256: fileHash(resultFile)};
    assert.throws(() => windowsGpuHostSdkPins(root, repinned), /metadata differs from its producer pins/);
  });

  test("wires three same-run inputs to one bounded build-only factory artifact", () => {
    const workflow = fs.readFileSync(path.join(repo, ".github/workflows/build-native-prerequisites.yml"), "utf8"),
      job = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-gpu-job.yml"), "utf8");
    assert.match(job, /runs-on: ubuntu-24\.04/); assert.match(job, /timeout-minutes: 60/);
    const deadline = job.indexOf("MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH=$(( $(date -u +%s) + 3300 ))"),
      provision = job.indexOf("Install SDK extraction tools and the selected compiler"), firstDownload = job.indexOf("uses: actions/download-artifact@");
    assert.ok(deadline >= 0 && deadline < provision && deadline < firstDownload, "the aggregate deadline begins before provisioning and downloads");
    const dependencies = job.indexOf("npm ci --ignore-scripts --no-audit --no-fund");
    assert.ok(dependencies >= 0 && dependencies < job.indexOf("run: node --test"),
      "SDK config tests load the controller helpers and need locked build dependencies first");
    /** @param {string} content */
    const expression = content => String.fromCharCode(36) + "{{ " + content + " }}";
    for (const field of ["bundle", "result", "config", "handoff"]) {
      assert.ok(job.includes("value: " + expression("jobs.factory.outputs." + field + "_sha256")),
        "reusable job exports the external " + field + " pin");
      assert.ok(job.includes(field + "_sha256: " + expression("steps.factory.outputs.windows_gpu_" + field + "_sha256")),
        "producer " + field + " pin reaches the reusable output unchanged");
    }
    for (const [name, profile] of [["build-windows-oneapi-gpu", "oneapi"], ["build-windows-opencl-gpu", "dpcpp-opencl"]]) {
      const caller = workflow.split("\n  " + name + ":\n")[1]?.split(/\n {2}[a-z][a-z0-9-]*:\n/)[0];
      assert.ok(caller, "distinct caller exists: " + name);
      assert.match(caller, /needs: \[linux-sources, provision-linux-dpcpp-cuda-inputs, install-audit-linux-sdk\]/);
      assert.match(caller, /uses: \.\/\.github\/workflows\/build-windows-gpu-job\.yml/);
      assert.ok(caller.includes("profile: " + profile));
      for (const [key, reference] of [
        ["source_bundle_sha256", "needs.linux-sources.outputs.bundle_sha256"],
        ["dpcpp_bundle_sha256", "needs.provision-linux-dpcpp-cuda-inputs.outputs.bundle_sha256"],
        ["dpcpp_manifest_sha256", "needs.provision-linux-dpcpp-cuda-inputs.outputs.manifest_sha256"],
        ["tools_bundle_sha256", "needs.install-audit-linux-sdk.outputs.tools_bundle_sha256"],
        ["tools_handoff_sha256", "needs.install-audit-linux-sdk.outputs.tools_handoff_sha256"],
      ]) {assert.ok(caller.includes(key + ": " + expression(reference || "")));}
    }
    for (const variable of ["SOURCE_BUNDLE", "DPCPP_BUNDLE", "DPCPP_MANIFEST", "TOOLS_BUNDLE", "TOOLS_HANDOFF"]) {
      assert.ok(job.includes("MOM_NATIVE_WINDOWS_GPU_" + variable + "_SHA256: " + expression("inputs." + variable.toLowerCase() + "_sha256")));
    }
    assert.equal((job.match(/uses: actions\/download-artifact@/g) || []).length, 8,
      "source, DPC++, tools, CUDA images, ACPP source/SDK, LLVM frontend and HIP headers are profile-selected");
    assert.match(job, /if: inputs\.profile == 'acpp-hip'\n {8}uses: actions\/download-artifact@/);
    assert.match(job, /if: inputs\.profile != 'acpp-hip'\n {8}uses: actions\/download-artifact@/);
    assert.match(job, /if: inputs\.profile == 'dpcpp' \|\| inputs\.profile == 'acpp-cuda'\n {8}uses: actions\/download-artifact@/);
    assert.equal((job.match(/if: startsWith\(inputs\.profile, 'acpp-'\)/g) || []).length, 3);
    assert.equal((job.match(/uses: actions\/upload-artifact@/g) || []).length, 1, "only the resulting factory handoff is uploaded");
    for (const artifact of ["native-linux-sdk-state-", "native-linux-dpcpp-cuda-inputs-", "native-linux-llvm-tools-"]) {
      assert.ok(job.includes("name: " + artifact), "input tar contract is present: " + artifact);
    }
    assert.ok(job.includes("name: native-windows-gpu-" + expression("inputs.profile")) &&
      job.includes("path: " + expression("runner.temp") + "/native-windows-gpu-" + expression("inputs.profile")),
    "one output tar is retained under its same-run name");
    assert.match(job, /run: scripts\/run-bounded-native-ci\.sh node scripts\/run-native-windows-gpu-ci\.js "\$MOM_NATIVE_WINDOWS_GPU_PROFILE"/);
    assert.match(job, /if \[\[ "\$MOM_NATIVE_WINDOWS_GPU_PROFILE" == oneapi \]\]; then/);
    assert.doesNotMatch(job, /actions\/cache|self-hosted|nvidia-smi|\/dev\/nvidia/);
    assert.match(workflow.slice(0, workflow.indexOf("jobs:")), /MOM_AOT_BUILD_ONLY: "1"[\s\S]*MOM_BUILD_JOBS: "2"/);
  });

  test("selects a separate OpenCL factory and profile-scoped descriptor paths", t => {
    const f = fixture(t);
    process.env["MOM_NATIVE_WINDOWS_GPU_DIR"] = path.join("/tmp", "mom-native-windows-gpu-dpcpp-opencl-" + f.suffix);
    const plan = windowsGpuCiJobPreflight("dpcpp-opencl");
    assert.equal(plan.factory.profile, "dpcpp-opencl");
    assert.equal(plan.configSource, path.join(f.runnerTemp, "windows-gpu-dpcpp-opencl-config-" + f.suffix + ".json"));
    assert.equal(plan.sourceTar, f.files[0]);
    assert.equal(fs.existsSync(plan.factory.root), false);
  });

  test("binds the DPC++ producer to the externally pinned source tar", t => {
    const f = fixture(t), file = path.join(f.runnerTemp, "inputs.json"), sourceSha256 = "e".repeat(64),
      manifest = {source: {tarSha256: sourceSha256}};
    fs.writeFileSync(file, JSON.stringify(manifest), {flag: "wx"});
    assert.deepEqual(windowsGpuDpcppManifest(file, fileHash(file), sourceSha256), manifest);
    assert.throws(() => windowsGpuDpcppManifest(file, fileHash(file), "f".repeat(64)), /different source artifact/);
    assert.throws(() => windowsGpuDpcppManifest(file, "0".repeat(64), sourceSha256), /external pin/);
  });

  test("forwards the verified same-run manifest pin to the strict DPC++ auditor before retiring the pinned tar", t => {
    const f = fixture(t), ci = require("../scripts/prepare-native-ci"),
      state = {...ci.dpcppCudaInputsStatePaths({suffix: f.suffix, runnerTemp: f.runnerTemp}), manifest: path.join(f.runnerTemp, "inputs.json")},
      sourceSha256 = fileHash(f.files[0] || ""), manifest = {source: {tarSha256: sourceSha256}};
    fs.writeFileSync(state.manifest, JSON.stringify(manifest) + "\n", {flag: "wx"});
    const manifestSha256 = fileHash(state.manifest);
    const source = fs.readFileSync(path.join(repo, "scripts/run-native-windows-gpu-ci.js"), "utf8"),
      run = source.slice(source.indexOf("function runWindowsGpuCiJob"), source.indexOf("\nmodule.exports")),
      restoreAt = run.indexOf('restore(plan, plan.dpcppTar || "", pins["DPCPP_BUNDLE"] || "", "/tmp", dpcpp.name)'),
      admitAt = run.indexOf('admitWindowsGpuDpcppInputs(dpcpp, pins["DPCPP_MANIFEST"] || "", pins["SOURCE_BUNDLE"] || "")'),
      retireAt = run.indexOf('fs.unlinkSync(plan.dpcppTar || "")', admitAt);
    assert.ok(restoreAt >= 0 && admitAt > restoreAt && retireAt > admitAt,
      "the externally bundle-pinned archive must be restored before strict manifest admission and retired only afterward");
    let calls = 0;
    t.mock.method(ci, "auditDpcppCudaInputs", (/** @type {typeof state} */ actualState,
      /** @type {typeof manifest} */ actualManifest, /** @type {string | undefined} */ actualPin) => {
      calls++; assert.equal(actualState, state); assert.deepEqual(actualManifest, manifest);
      assert.equal(actualPin, manifestSha256, "forward the byte-verified external manifest pin unchanged");
      return actualManifest;
    });
    assert.deepEqual(admitWindowsGpuDpcppInputs(state, manifestSha256, sourceSha256), manifest);
    assert.equal(calls, 1);
    assert.throws(() => admitWindowsGpuDpcppInputs(state, "0".repeat(64), sourceSha256), /external pin/);
    assert.throws(() => admitWindowsGpuDpcppInputs(state, manifestSha256, "f".repeat(64)), /different source artifact/);
    assert.equal(calls, 1, "invalid same-run manifest/source pins must fail before the current auditor");
  });

  test("admits a separately pinned CUDA-image artifact only for the DPC++ factory", t => {
    const f = fixture(t);
    process.env["MOM_NATIVE_WINDOWS_GPU_DIR"] = path.join("/tmp", "mom-native-windows-gpu-dpcpp-" + f.suffix);
    assert.throws(() => windowsGpuCiJobPlan("dpcpp"), /external image-bundle SHA-256/);
    const archive = path.join(f.runnerTemp, "incoming", "native-linux-cuda-images-" + f.suffix + ".tar");
    fs.writeFileSync(archive, "original CUDA images", {flag: "wx"});
    process.env["MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256"] = fileHash(archive);
    const plan = windowsGpuCiJobPreflight("dpcpp");
    assert.equal(plan.cudaImages?.archive, archive);
    assert.equal(plan.cudaImages?.root, "/tmp/mom-native-cuda-images-handoff-" + f.suffix);
    fs.appendFileSync(archive, "changed");
    assert.throws(() => windowsGpuCiJobPreflight("dpcpp"), /differs from its external pin/);
    assert.equal(fs.existsSync(plan.factory.root), false);
    process.env["MOM_NATIVE_WINDOWS_GPU_DIR"] = path.join("/tmp", "mom-native-windows-gpu-oneapi-" + f.suffix);
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /Only the DPC\+\+ CUDA or ACPP CUDA factories/);
    const workflow = fs.readFileSync(path.join(repo, ".github/workflows/build-native-prerequisites.yml"), "utf8"),
      caller = workflow.split("\n  build-windows-dpcpp-gpu:\n")[1]?.split(/\n {2}[a-z][a-z0-9-]*:\n/)[0] || "";
    assert.match(caller, /needs: \[linux-sources, provision-linux-dpcpp-cuda-inputs, install-audit-linux-sdk, build-linux-cuda-images\]/);
    assert.ok(caller.includes("images_bundle_sha256: " + String.fromCharCode(36) + "{{ needs.build-linux-cuda-images.outputs.bundle_sha256 }}"));
  });

  test("admits ACPP CUDA only with the same-run ACPP, frontend, DPC++ and CUDA image bundles", t => {
    const f = fixture(t);
    addAcppArtifacts(f, "acpp-cuda");
    const plan = windowsGpuCiJobPreflight("acpp-cuda");
    assert.equal(plan.acpp?.sourceTar, path.join(f.runnerTemp, "incoming", "native-acpp-state-" + f.suffix + "-sources.tar"));
    assert.equal(plan.acpp?.sdkRoot, path.join("/tmp", "mom-native-acpp-cuda-hip-sdk-" + f.suffix));
    assert.equal(plan.acpp?.frontendRoot, path.join("/tmp", "native-linux-llvm-acpp-frontend-" + f.suffix));
    assert.equal(plan.dpcpp?.name, "mom-native-dpcpp-cuda-inputs-" + f.suffix);
    assert.equal(plan.cudaImages?.bundleSha256, process.env["MOM_NATIVE_WINDOWS_GPU_IMAGES_BUNDLE_SHA256"]);
    const composerInput = windowsGpuAcppConfigInput(plan, "/tmp/native-windows-host-sdk-" + f.suffix,
      {resultSha256: "a".repeat(64), descriptorSha256: "b".repeat(64), filesSha256: "c".repeat(64), overlaySha256: "d".repeat(64)});
    assert.equal(composerInput.acpp.sourceRoot, plan.acpp?.source.preparedCheckout);
    assert.notEqual(composerInput.acpp.sourceRoot, plan.acpp?.source.source,
      "the public composer receives prepared/acpp rather than the outer source-cohort root");
    assert.ok(f.files.every(file => fs.existsSync(file)));
    for (const key of ["ACPP_SOURCE_BUNDLE", "ACPP_SDK_BUNDLE", "FRONTEND_BUNDLE", "FRONTEND_HANDOFF", "DPCPP_BUNDLE", "DPCPP_MANIFEST"]) {
      assert.match(plan.pins[key] || "", /^[a-f0-9]{64}$/);
    }
    const frontendPin = process.env["MOM_NATIVE_WINDOWS_GPU_FRONTEND_HANDOFF_SHA256"];
    process.env["MOM_NATIVE_WINDOWS_GPU_FRONTEND_HANDOFF_SHA256"] = "";
    assert.throws(() => windowsGpuCiJobPlan("acpp-cuda"), /every selected input's external same-run SHA-256 pin/);
    process.env["MOM_NATIVE_WINDOWS_GPU_FRONTEND_HANDOFF_SHA256"] = frontendPin;
    assert.equal(fs.existsSync(plan.factory.root), false);
  });

  test("keeps ACPP HIP independent of DPC++/CUDA inputs and uses only the system loader path", t => {
    const f = fixture(t);
    addAcppArtifacts(f, "acpp-hip");
    const plan = windowsGpuCiJobPreflight("acpp-hip");
    assert.equal(plan.acpp?.sdkRoot, path.join("/tmp", "mom-native-acpp-cuda-hip-sdk-" + f.suffix));
    assert.equal(plan.dpcpp, null);
    assert.equal(plan.dpcppTar, null);
    assert.equal(plan.cudaImages, null);
    assert.equal(plan.hipHeaders?.root, "/tmp/native-rocm-hip-headers-" + f.suffix);
    const composed = windowsGpuAcppConfigInput(plan, "/tmp/sdk", /** @type {any} */ ({}));
    assert.deepEqual(composed.hipHeaders, {root: plan.hipHeaders?.root, handoffSha256: "e".repeat(64)});
    assert.equal(Object.hasOwn(plan.pins, "DPCPP_BUNDLE"), false);
    assert.equal(windowsGpuLibraryPath("acpp-hip", null), "/usr/lib/x86_64-linux-gnu");
    assert.equal(windowsGpuLibraryPath("acpp-cuda", null), "/usr/lib/x86_64-linux-gnu");
    assert.throws(() => windowsGpuLibraryPath("dpcpp", null), /require their admitted compiler root/);

    const workflow = fs.readFileSync(path.join(repo, ".github/workflows/build-native-prerequisites.yml"), "utf8"),
      cudaCaller = workflow.split("\n  build-windows-acpp-cuda-gpu:\n")[1]?.split(/\n {2}[a-z][a-z0-9-]*:\n/)[0] || "",
      hipCaller = workflow.split("\n  build-windows-acpp-hip-gpu:\n")[1]?.split(/\n {2}[a-z][a-z0-9-]*:\n/)[0] || "";
    assert.match(cudaCaller, /provision-linux-dpcpp-cuda-inputs/);
    assert.match(cudaCaller, /build-linux-cuda-images/);
    for (const key of ["acpp_source_bundle_sha256", "acpp_sdk_bundle_sha256", "frontend_bundle_sha256", "frontend_handoff_sha256"]) {
      assert.ok(cudaCaller.includes(key + ": "), "ACPP CUDA caller pins " + key);
      assert.ok(hipCaller.includes(key + ": "), "ACPP HIP caller pins " + key);
    }
    assert.doesNotMatch(hipCaller, /provision-linux-dpcpp-cuda-inputs|build-linux-cuda-images|dpcpp_bundle_sha256|dpcpp_manifest_sha256|images_bundle_sha256/);
    assert.match(hipCaller, /build-linux-rocm-comgr-sdk/);
    for (const key of ["hip_headers_bundle_sha256", "hip_headers_handoff_sha256"]) {assert.ok(hipCaller.includes(key + ": "));}
  });

  test("requires both HIP header pins and rejects unused HIP inputs on CUDA", t => {
    const f = fixture(t); addAcppArtifacts(f, "acpp-hip");
    delete process.env["MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256"];
    assert.throws(() => windowsGpuCiJobPreflight("acpp-hip"), /every selected input's external same-run SHA-256 pin/);
    process.env["MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256"] = "e".repeat(64);
    const plan = windowsGpuCiJobPreflight("acpp-hip");
    fs.appendFileSync(plan.hipHeaders?.archive || "", "changed");
    assert.throws(() => windowsGpuCiJobPreflight("acpp-hip"), /differs from its external pin/);
    assert.equal(fs.existsSync(plan.factory.root), false);
  });

  test("rejects unused HIP header pins for other profiles", t => {
    fixture(t); process.env["MOM_NATIVE_WINDOWS_GPU_HIP_HEADERS_HANDOFF_SHA256"] = "e".repeat(64);
    assert.throws(() => windowsGpuCiJobPlan("oneapi"), /unused by its selected profile/);
  });

  test("exports exact original CUDA and ACPP bundle pins and fails closed before publishing missing or redirected inputs", t => {
    const f = fixture(t), workflow = fs.readFileSync(path.join(repo, ".github/workflows/build-native-prerequisites.yml"), "utf8");
    for (const [job, basename] of /** @type {Array<[string,string]>} */ ([
      ["fetch-linux-acpp-sources", "native-acpp-state-123-1-sources.tar"],
      ["build-linux-cuda-images", "native-linux-cuda-images-123-1.tar"],
      ["build-linux-acpp-cuda-hip-sdk", "native-linux-acpp-cuda-hip-sdk-123-1.tar"],
    ])) {
      const body = workflow.split("\n  " + job + ":\n")[1]?.split(/\n {2}[a-z][a-z0-9-]*:\n/)[0] || "";
      assert.ok(body.includes("bundle_sha256: " + String.fromCharCode(36) + "{{ steps.handoff.outputs.bundle_sha256 }}"));
      const script = body.match(/ {8}id: handoff\n {8}run: \|\n((?: {10}.*\n)+)/)?.[1]?.replace(/^ {10}/gm, "") || "";
      assert.ok(script, "the bundle pin is emitted by a dedicated post-producer step");
      const archive = path.join(f.runnerTemp, basename), env = {...process.env, RUNNER_TEMP: f.runnerTemp,
        GITHUB_RUN_ID: "123", GITHUB_RUN_ATTEMPT: "1", GITHUB_OUTPUT: f.output};
      const run = () => spawnSync("/bin/bash", ["-euo", "pipefail", "-c", script], {env, encoding: "utf8"});
      assert.notEqual(run().status, 0, "a missing producer archive must not publish an empty pin");
      assert.equal(fs.readFileSync(f.output, "utf8"), "");
      fs.writeFileSync(archive, "original " + job, {flag: "wx"});
      assert.equal(run().status, 0);
      assert.equal(fs.readFileSync(f.output, "utf8"), "bundle_sha256=" + fileHash(archive) + "\n");
      fs.writeFileSync(f.output, ""); fs.unlinkSync(archive); fs.symlinkSync(f.files[0] || "", archive);
      assert.notEqual(run().status, 0, "a symlink cannot substitute a different handoff");
      assert.equal(fs.readFileSync(f.output, "utf8"), ""); fs.unlinkSync(archive);
    }
  });
});
