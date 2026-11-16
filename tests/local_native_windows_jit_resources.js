"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const local = require("../scripts/build-local-native-sdk"), {fileHash, buildFileInputs, sourceInputs} = require("../scripts/build-source-inputs"),
  {localWindowsJitResourcePaths, buildLocalWindowsJitResources} = require("../scripts/build-local-native-windows-jit-resources"),
  {jitSourceInputs} = require("../scripts/native-jit-sdk");

const repo = path.resolve(__dirname, ".."), linuxTarget = "x86_64-unknown-linux-gnu", windowsTarget = "x86_64-w64-windows-gnu";

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const workspace = fs.mkdtempSync(path.join("/tmp", "mom-local-windows-jit-")), support = local.supportPaths(workspace),
    linuxRoot = support.linuxJitResources, baseRoot = path.join(linuxRoot, "base", "base"), baseManifest = path.join(linuxRoot, "base", "base-manifest.json"),
    dpcppRoot = path.join(support.dpcppCudaRoot, "dpcpp"), cudaRoot = path.join(support.dpcppCudaRoot, "cuda"),
    manifest = path.join(support.dpcppCudaRoot, "inputs.json"), sourceOrigin = {origin: "workspace-prepared-receipts", workspaceRoot: workspace},
    record = {version: 1, target: "fixture-local-dpcpp-input", buildOnly: true, source: sourceOrigin};
  fs.mkdirSync(path.join(workspace, "sources"));
  fs.writeFileSync(path.join(workspace, ".mom-native-sdk-workspace.json"), JSON.stringify({version: 1, checkout: fs.realpathSync(repo)}) + "\n");
  fs.mkdirSync(path.join(dpcppRoot, "include/sycl"), {recursive: true}); fs.mkdirSync(path.join(dpcppRoot, "bin"), {recursive: true});
  fs.mkdirSync(path.join(cudaRoot, "include"), {recursive: true});
  fs.writeFileSync(path.join(dpcppRoot, "include/sycl/sycl.hpp"), "fixture SYCL header\n");
  fs.writeFileSync(manifest, JSON.stringify(record) + "\n");
  fs.mkdirSync(path.join(baseRoot, "include"), {recursive: true});
  fs.mkdirSync(path.join(baseRoot, "lib/clang/23/include"), {recursive: true});
  fs.writeFileSync(baseManifest, JSON.stringify({version: 1, files: []}) + "\n");
  fs.mkdirSync(path.join(linuxRoot, "resources"), {recursive: true});
  fs.writeFileSync(path.join(linuxRoot, "resources/manifest.json"), "{}\n");

  const toolsRoot = path.join(workspace, "fixture-tools"), tools = /** @type {{cmake:string,ninja:string,cc:string,cxx:string}} */ ({
    cmake: path.join(toolsRoot, "cmake"), ninja: path.join(toolsRoot, "ninja"), cc: path.join(toolsRoot, "icx"), cxx: path.join(toolsRoot, "icpx"),
  });
  fs.mkdirSync(toolsRoot);
  for (const file of Object.values(tools)) {fs.writeFileSync(file, "#!/bin/sh\nexit 0\n"); fs.chmodSync(file, 0o755);}

  const localSdk = {workspace, compilerFetch: support.compilerFetch, compilerSources: support.compilerSources,
      buildInputs: support.buildInputs, llvmSdk: support.llvmSdk}, counts = {dpcpp: 0, linux: 0, gnuBuild: 0, gnuAudit: 0,
      emitterBuild: 0, resourceBuild: 0, resourceAudit: 0, scope: 0};
  let driftLinuxAudit = false;
  /** @param {{root:string}} state @param {typeof windowsTarget} target */
  const buildGnu = (state, target) => {
    counts.gnuBuild++;
    assert.equal(target, windowsTarget);
    fs.mkdirSync(state.root, {recursive: false});
    const header = path.join(state.root, "include/mom-system-target/windows.h"), notice = path.join(state.root, "licenses/mingw-w64-common/copyright");
    fs.mkdirSync(path.dirname(header), {recursive: true}); fs.mkdirSync(path.dirname(notice), {recursive: true});
    fs.writeFileSync(header, "fixture Windows target header\n"); fs.writeFileSync(notice, "fixture package terms\n");
    fs.writeFileSync(path.join(state.root, "target-headers.json"), JSON.stringify([header]) + "\n");
    fs.writeFileSync(path.join(state.root, "result.json"), "fixture GNU receipt\n");
    fs.writeFileSync(path.join(state.root, "copyrights.json"), "fixture copyright receipt\n");
    return /** @type {any} */ (auditGnu({root: state.root}, windowsTarget));
  };
  /** @param {{root:string,target?:typeof windowsTarget}} state @param {typeof windowsTarget} target */
  const auditGnu = (state, target) => {
    counts.gnuAudit++;
    assert.equal(target, windowsTarget);
    const root = state.root, header = path.join(root, "include/mom-system-target/windows.h"), noticePath = path.join(root, "licenses/mingw-w64-common/copyright");
    for (const file of [path.join(root, "result.json"), path.join(root, "copyrights.json"), path.join(root, "target-headers.json"), header, noticePath]) {
      assert.ok(fs.lstatSync(file).isFile());
    }
    return {headers: {root, target: windowsTarget, files: [{source: header, virtual: "include/mom-system-target/windows.h",
      package: "mingw-w64-common", sha256: fileHash(header), bytes: fs.statSync(header).size}]},
    notices: [{package: "mingw-w64-common", path: noticePath, sha256: fileHash(noticePath)}]};
  };
  /** @param {string} sdk @param {string} output @param {NodeJS.ProcessEnv} environment @param {string[]} targetHeaders */
  const buildEmitter = (sdk, output, environment, targetHeaders) => {
    counts.emitterBuild++;
    assert.equal(environment["MOM_AOT_BUILD_ONLY"], "1"); assert.equal(environment["MOM_BUILD_JOBS"], "2");
    assert.equal(environment["OMP_NUM_THREADS"], "2");
    fs.mkdirSync(output);
    const compiler = path.join(sdk, "bin/clang++"), fixtureFile = path.join(output, "native-kawpow-source"), sycl = path.join(sdk, "include/sycl/sycl.hpp");
    fs.writeFileSync(compiler, "fixture DPC++ clang++\n"); fs.chmodSync(compiler, 0o755);
    fs.writeFileSync(fixtureFile, "fixture emitter executable\n");
    const leaves = [...new Set([...jitSourceInputs, "scripts/native-jit-sdk.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"])],
      inputs = [process.execPath, compiler, sycl, ...targetHeaders, ...leaves.map(name => path.join(repo, name))].map(file => fs.realpathSync(file)),
      receipt = {version: 1, fixture: fixtureFile, fixtureSha256: fileHash(fixtureFile), compiler, compilerSha256: fileHash(compiler),
        inputs: buildFileInputs(inputs), recipe: sourceInputs(repo, [], leaves), targetHeaders,
        buildOnly: true, executed: false, publicationQualified: false, scope: "fixture-only emitter, not compiled"};
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(receipt, null, 2) + "\n");
    return receipt;
  };
  const dependencies = /** @type {NonNullable<Parameters<typeof buildLocalWindowsJitResources>[2]>} */ (/** @type {unknown} */ ({
    environment: {PATH: "/usr/bin:/bin"}, verifyScope() {counts.scope++;},
    /** @param {{cmake:string,ninja:string,cc:string,cxx:string}} selected */ validateTools(selected) {local.localJitToolsReceipt(selected);},
    /** @param {import("../scripts/native-dpcpp-cuda-inputs").DpcppCudaInputState} state @param {any} admitted */
    auditDpcpp(state, admitted) {
      counts.dpcpp++;
      assert.equal(state.workspaceRoot, workspace); assert.equal(admitted.source.origin, "workspace-prepared-receipts");
      assert.equal(arguments.length, 2, "local source provenance must not be treated as an external CI manifest pin");
      return admitted;
    },
    /** @param {import("../scripts/prepare-native-ci").LinuxJitResourceTreeState} state
     * @param {{cmake:string,ninja:string,cc:string,cxx:string}} selectedTools @param {NodeJS.ProcessEnv} env */
    auditLinux(state, selectedTools, env) {
      counts.linux++;
      assert.equal(state.root, linuxRoot); assert.equal(selectedTools, tools); assert.equal(arguments.length, 3);
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1"); assert.equal(env["MOM_BUILD_JOBS"], "2");
      const sourceDate = driftLinuxAudit && counts.linux > 1 ? "20261115" : "20261114";
      const resourceRoot = path.join(linuxRoot, "resources"), resourceManifest = path.join(resourceRoot, "manifest.json"), manifestHash = fileHash(resourceManifest);
      return /** @type {any} */ ({resources: {target: linuxTarget, directory: resourceRoot, overlay: path.join(resourceRoot, "overlay"), manifestHash,
        inputs: {[resourceManifest]: manifestHash}},
      sourceDate, fingerprints: {[manifest]: fileHash(manifest)}});
    },
    buildGnu, auditGnu, buildEmitter,
    /** @param {unknown} _profile @param {string} output */
    prepareResources(_profile, output) {
      counts.resourceBuild++; fs.mkdirSync(output);
      fs.writeFileSync(path.join(output, "manifest.json"), "fixture Windows JIT resource manifest\n");
    },
    /** @param {string} root @param {string} target */
    verifyResources(root, target) {
      counts.resourceAudit++;
      assert.equal(target, windowsTarget);
      return {target, directory: root, manifestHash: fileHash(path.join(root, "manifest.json"))};
    },
  }));
  t.after(() => fs.rmSync(workspace, {recursive: true, force: true}));
  return {workspace, support, paths: localWindowsJitResourcePaths(workspace), localSdk, tools, dependencies, counts,
    /** @param {boolean} value */ setLinuxAuditDrift(value) {driftLinuxAudit = value;}, sourceOrigin, record, manifest};
}

test.describe("Local Linux-hosted Windows source-JIT resources", {concurrency: false, skip: process.platform !== "linux"}, () => {
  test("uses stable workspace paths without a CI run suffix", () => {
    const paths = localWindowsJitResourcePaths("/tmp/mom-local-workspace");
    assert.equal(paths.root, "/tmp/mom-local-workspace/windows-jit-resources");
    assert.equal(paths.gnu, path.join(paths.root, "windows-gnu-headers"));
    assert.equal(paths.resources, path.join(paths.root, "resources"));
    assert.throws(() => localWindowsJitResourcePaths("relative-workspace"), /canonical and absolute/);
  });

  test("checks the memory scope before validating SDK context or tools", t => {
    const selected = fixture(t); let toolsChecked = false;
    assert.throws(() => buildLocalWindowsJitResources(null, selected.tools, {
      verifyScope() {throw new Error("fixture memory scope missing");}, validateTools() {toolsChecked = true;},
    }), /fixture memory scope missing/);
    assert.equal(toolsChecked, false);
  });

  test("builds once, retains original receipts and reuses only after full re-admission", t => {
    const selected = fixture(t), first = buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies),
      countsAfterBuild = {...selected.counts}, second = buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies);
    assert.equal(first.reused, false); assert.equal(second.reused, true);
    assert.equal(first.target, windowsTarget); assert.equal(first.buildOnly, true); assert.equal(first.executed, false);
    assert.equal(first.publicationQualified, false); assert.equal(first.dpcpp.sourceOrigin.origin, "workspace-prepared-receipts");
    assert.equal(first.profile.record.target, windowsTarget);
    assert.equal(JSON.stringify(first).includes("runAttempt"), false);
    assert.equal(selected.counts.gnuBuild, 1); assert.equal(selected.counts.emitterBuild, 1); assert.equal(selected.counts.resourceBuild, 1);
    assert.ok(selected.counts.dpcpp > countsAfterBuild.dpcpp); assert.ok(selected.counts.linux > countsAfterBuild.linux);
    assert.ok(selected.counts.gnuAudit > countsAfterBuild.gnuAudit); assert.ok(selected.counts.resourceAudit > countsAfterBuild.resourceAudit);
  });

  test("rejects changed local DPC++ source provenance before creating managed output", t => {
    const selected = fixture(t), manifest = JSON.parse(fs.readFileSync(selected.manifest, "utf8"));
    manifest.source = {origin: "source-tar", workspaceRoot: selected.workspace};
    fs.writeFileSync(selected.manifest, JSON.stringify(manifest) + "\n");
    assert.throws(() => buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies), /original prepared-receipt DPC\+\+ source provenance/);
    assert.equal(fs.existsSync(selected.paths.root), false); assert.equal(selected.counts.dpcpp, 0); assert.equal(selected.counts.gnuBuild, 0);
  });

  test("preserves partial or unknown managed output instead of repairing it", t => {
    const selected = fixture(t); fs.mkdirSync(selected.paths.root); fs.writeFileSync(path.join(selected.paths.root, "unknown"), "preserve\n");
    assert.throws(() => buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies), /incomplete or contains unknown state/);
    assert.equal(fs.readFileSync(path.join(selected.paths.root, "unknown"), "utf8"), "preserve\n");
    assert.equal(selected.counts.dpcpp, 0); assert.equal(selected.counts.gnuBuild, 0);
  });

  test("does not return a handoff if the original local input graph changes during production", t => {
    const selected = fixture(t); selected.setLinuxAuditDrift(true);
    assert.throws(() => buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies), /changed during producer admission/);
    assert.equal(fs.existsSync(selected.paths.root), true);
    assert.equal(selected.counts.gnuBuild, 1); assert.equal(selected.counts.emitterBuild, 1);
  });

  test("rejects a stale cached profile without rebuilding or replacing preserved outputs", t => {
    const selected = fixture(t); buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies);
    const before = {...selected.counts}; fs.writeFileSync(selected.paths.profile, "{}\n");
    assert.throws(() => buildLocalWindowsJitResources(selected.localSdk, selected.tools, selected.dependencies), /profile differs from its admitted producer inputs/);
    assert.deepEqual({gnuBuild: selected.counts.gnuBuild, emitterBuild: selected.counts.emitterBuild, resourceBuild: selected.counts.resourceBuild},
      {gnuBuild: before.gnuBuild, emitterBuild: before.emitterBuild, resourceBuild: before.resourceBuild});
    assert.equal(fs.readFileSync(selected.paths.profile, "utf8"), "{}\n");
  });

  test("preserves producer output when final resource admission fails", t => {
    const selected = fixture(t), dependencies = /** @type {NonNullable<Parameters<typeof buildLocalWindowsJitResources>[2]>} */ ({...selected.dependencies,
      verifyResources() {throw new Error("fixture resource admission failed");}});
    assert.throws(() => buildLocalWindowsJitResources(selected.localSdk, selected.tools, dependencies), /fixture resource admission failed/);
    assert.equal(fs.existsSync(selected.paths.root), true);
    assert.equal(fs.existsSync(path.join(selected.paths.root, "resource-profile.json")), true);
    assert.equal(fs.existsSync(path.join(selected.paths.root, "resources")), true);
  });
});
