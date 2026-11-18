"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"),
  test = require("node:test"), {runInNewContext, Script} = require("node:vm"), {fileHash} = require("../scripts/build-source-inputs");
const workflow = fs.readFileSync(path.join(__dirname, "../.github/workflows/pack-native-application.yml"), "utf8"),
  bodies = [...workflow.matchAll(/node <<'NODE'\n([\s\S]*?)\n {10}NODE/g)].map(match =>
    (match[1] || "").split("\n").map(line => line.slice(10)).join("\n"));

/** Exercise the maintained job body without network, compiler or miner execution. The
 * existing packNativeEntrypoint suite and real-tool smoke separately cover the packing API.
 * @param {import('node:test').TestContext} t @param {'linux'|'win32'} [target] */
function fixture(t, target = "linux") {
  const temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-packing-ci-")), suffix = "123456789-1",
    platform = target === "win32" ? "windows" : "linux", binary = target === "win32" ? "mom.exe" : "mom",
    metadataName = `native-${platform}-application-build-${suffix}`, metadata = path.join("/tmp", metadataName),
    output = path.join("/tmp", `native-${platform}-packed-build-${suffix}`), calls = /** @type {any[]} */ ([]);
  // This harness never creates the fixed /tmp roots: the virtual FS maps only those
  // two exact roots into this test's private temporary directory.
  const mapped = (/** @type {string} */ file) => file === metadata || file.startsWith(metadata + path.sep) ?
    path.join(temp, "metadata", path.relative(metadata, file)) : file === output || file.startsWith(output + path.sep) ?
      path.join(temp, "packed", path.relative(output, file)) : file;
  const virtualFs = new Proxy(fs, {get(object, key) {
    const value = Reflect.get(object, key);
    if (typeof value !== "function") {return value;}
    if (key === "copyFileSync") {return (/** @type {string} */ from, /** @type {string} */ to, /** @type {number} */ flags) => fs.copyFileSync(mapped(from), mapped(to), flags);}
    if (key === "realpathSync") {return (/** @type {string} */ file) => {const real = fs.realpathSync(mapped(file)); return real === mapped(file) ? file : real;};}
    return (/** @type {string} */ file, /** @type {any[]} */ ...args) => value.call(object, typeof file === "string" ? mapped(file) : file, ...args);
  }});
  const rawDirectory = path.join(temp, "native-raw"), download = path.join(temp, "native-metadata");
  fs.mkdirSync(rawDirectory); fs.mkdirSync(download); fs.mkdirSync(path.join(temp, "metadata"));
  const raw = path.join(rawDirectory, binary); fs.writeFileSync(raw, "own fixture executable");
  const archive = path.join(download, metadataName + ".tar"); fs.writeFileSync(archive, "fixture transport");
  const save = (/** @type {string} */ name, /** @type {any} */ value) => fs.writeFileSync(path.join(temp, "metadata", name), JSON.stringify(value));
  save("application-config.json", {buildOnly: true}); save("result.json", {buildOnly: true});
  save("entrypoint.json", {configHash: fileHash(path.join(temp, "metadata/application-config.json"))});
  const handoff = {version: 1, format: `mom-${platform}-application-build-handoff-v1`, runAttempt: suffix,
    target: target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu",
    artifact: {file: binary, bytes: fs.statSync(raw).size, sha256: fileHash(raw)},
    config: {file: "application-config.json", sha256: fileHash(path.join(temp, "metadata/application-config.json"))},
    producerReceipt: {file: "result.json", sha256: fileHash(path.join(temp, "metadata/result.json"))},
    entrypointReceipt: {file: "entrypoint.json", sha256: fileHash(path.join(temp, "metadata/entrypoint.json"))},
    buildOnly: true, executed: false, publicationQualified: false};
  save("handoff.json", handoff);
  const env = {RUNNER_TEMP: temp, GITHUB_RUN_ID: "123456789", GITHUB_RUN_ATTEMPT: "1", NATIVE_PLATFORM: target,
    NATIVE_RAW_ARTIFACT: `native-${platform}-executable-${suffix}`, NATIVE_RAW_SHA256: fileHash(raw),
    NATIVE_METADATA_ARTIFACT: metadataName + ".tar", NATIVE_METADATA_SHA256: fileHash(archive),
    NATIVE_HANDOFF_SHA256: fileHash(path.join(temp, "metadata/handoff.json")),
    GITHUB_OUTPUT: path.join(temp, "outputs"), GITHUB_STEP_SUMMARY: path.join(temp, "summary")};
  const tool = JSON.parse(JSON.stringify(require("../scripts/native-packing-tool.json"))),
    hash = (/** @type {string} */ file) => fileHash(mapped(file));
  // Tool bytes are deliberately inert fixture data, never executable host programs.
  const toolArchive = "fixture upx archive", toolBytes = "fixture upx";
  const {createHash} = require("node:crypto"), digest = (/** @type {string} */ value) => createHash("sha256").update(value).digest("hex");
  tool.archive.bytes = Buffer.byteLength(toolArchive); tool.archive.sha256 = digest(toolArchive);
  tool.executable.bytes = Buffer.byteLength(toolBytes); tool.executable.sha256 = digest(toolBytes);
  let members = [metadataName, ...["application-config.json", "entrypoint.json", "handoff.json", "result.json"].map(name => metadataName + "/" + name)];
  const ci = {listState: () => members, extractState: () => calls.push({stage: "restore"}),
    packState: (/** @type {string} */ file) => {fs.writeFileSync(file, "fixture packed transport"); calls.push({stage: "seal"});}};
  /** @type {typeof import('../scripts/prepare-native-packing-tool') | undefined} */ let packingHelper;
  const execute = (/** @type {number} */ index) => runInNewContext(bodies[index] || "", {process: {env}, console: {log() { /* Captured fixture output is intentionally quiet. */ }}, require: function sandboxRequire(/** @type {string} */ name) {
    if (name === "node:fs") {return virtualFs;}
    if (name === "./scripts/prepare-native-ci") {return ci;}
    if (name === "./scripts/build-source-inputs") {return {fileHash: hash, readBuildJson: (/** @type {string} */ file) => JSON.parse(fs.readFileSync(mapped(file), "utf8"))};}
    if (name === "./scripts/native-packing-tool.json") {return tool;}
    if (name === "./scripts/prepare-native-packing-tool") {
      const module = {exports: {}};
      runInNewContext(fs.readFileSync(path.join(__dirname, "../scripts/prepare-native-packing-tool.js"), "utf8"), {
        module, process: {platform: "linux", arch: "x64", env},
        require: (/** @type {string} */ dependency) => sandboxRequire(dependency.startsWith("./") ? "./scripts/" + dependency.slice(2) : dependency),
      });
      packingHelper = /** @type {typeof import('../scripts/prepare-native-packing-tool')} */ (module.exports);
      return packingHelper;
    }
    if (name === "node:child_process") {
      return {spawnSync: (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {any} */ options) => {
        calls.push({stage: command, args});
        if (command === "curl") {fs.writeFileSync(args[args.indexOf("--output") + 1] || "", toolArchive);} else if (command === "tar") {fs.writeSync(options.stdio[1], toolBytes);} else {throw new Error("Unexpected external command: " + command);}
        return {status: 0};
      }};
    }
    if (name === "./scripts/build-release") {
      return {packNativeEntrypoint: (/** @type {any} */ options) => {
        calls.push({stage: "pack", options}); assert.equal(options.entrypointSha256, handoff.entrypointReceipt.sha256);
        assert.equal(hash(options.producerReceiptPath), handoff.producerReceipt.sha256);
        fs.mkdirSync(mapped(output)); fs.writeFileSync(mapped(path.join(output, binary)), "packed fixture");
        return {artifact: {sha256: hash(path.join(output, binary))}};
      }};
    }
    return require(name);
  }}, {timeout: 10000});
  t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
  return {env, calls, execute, handoff, save, raw, archive, temp,
    preparePacking: () => {
      assert.ok(packingHelper);
      return packingHelper.prepareNativePackingTool(path.join(temp, "native-packing-tool"), env);
    }, members: (/** @type {string[]} */ value) => {members = value;},
    reseal: () => {save("handoff.json", handoff); env.NATIVE_HANDOFF_SHA256 = fileHash(path.join(temp, "metadata/handoff.json"));}};
}

test.describe("SDK-free native packing CI handoff", {skip: process.platform !== "linux" && "Linux workflow harness"}, () => {
  test("job has bounded time, pinned actions and no SDK build, cache or miner execution", () => {
    assert.equal(bodies.length, 2); bodies.forEach(body => new Script(body));
    assert.match(workflow, /timeout-minutes: 30/); assert.match(workflow, /timeout-minutes: 25/);
    assert.doesNotMatch(workflow, /actions\/cache@|npm run build|npm ci|self-hosted/);
    for (const match of workflow.matchAll(/uses: (actions\/[^\s@]+)@([^\s#]+)/g)) {assert.match(match[2] || "", /^[a-f0-9]{40}$/);}
    assert.match(workflow, /archive: true/); assert.match(workflow, /compression-level: 0/);
  });
  test("packing failures upload only this attempt's exact UPX stage logs", () => {
    const block = workflow.split("      - name: Upload UPX stage logs after a packing failure\n")[1]?.split("\n      - name:")[0] || "",
      expression = String.fromCharCode(36, 123, 123), platform = expression + " inputs.platform == 'win32' && 'windows' || 'linux' }}",
      suffix = expression + " github.run_id }}-" + expression + " github.run_attempt }}",
      directory = `/tmp/native-${platform}-packed-build-${suffix}`,
      paths = block.split("          path: |\n")[1]?.split("\n          if-no-files-found:")[0]?.split("\n").map(line => line.trim()) || [];
    assert.match(block, /if: failure\(\) && steps\.pack\.outcome == 'failure'/);
    assert.match(block, /uses: actions\/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9/);
    assert.ok(block.includes(`name: native-${platform}-packing-diagnostics-${suffix}`));
    assert.deepEqual(paths, ["pack.log", "integrity.log", "restore.log"].map(name => `${directory}/${name}`));
    assert.match(block, /if-no-files-found: ignore/); assert.match(block, /retention-days: 1/);
    assert.doesNotMatch(block, /\*|mom\.exe|\/mom(?:\s|$)|result\.json|entrypoint\.json/);
    assert.equal(bodies.length, 2, "failure diagnostics do not add another executable workflow body");
  });
  test("official packer and its source notice cohort remain independently pinned", () => {
    const pin = require("../scripts/native-packing-tool.json");
    assert.equal(pin.version, "5.2.1"); assert.equal(pin.archive.bytes, 664048); assert.equal(pin.source.bytes, 1778024);
    for (const archive of [pin.archive, pin.source]) {
      assert.match(archive.url, /^https:\/\/github\.com\/upx\/upx\/releases\/download\/v5\.2\.1\/upx-5\.2\.1-/);
      assert.match(archive.sha256, /^[a-f0-9]{64}$/);
    }
    assert.deepEqual(pin.source.notices.map(row => row.member), ["upx-5.2.1-src/LICENSE", "upx-5.2.1-src/COPYING", "upx-5.2.1-src/vendor/lzma-sdk/LICENSE.txt"]);
    for (const [index, row] of pin.source.notices.entries()) {
      assert.match(row.sha256, /^[a-f0-9]{64}$/);
      if (index < 2) {assert.equal(row.sha256, pin.notices[index]?.sha256, "source and official binary distribute identical license texts");}
    }
  });
  test("each platform packs once and tests the exact packed artifact independently of raw tests", () => {
    const graph = fs.readFileSync(path.join(__dirname, "../.github/workflows/build-native-prerequisites.yml"), "utf8");
    for (const platform of ["linux", "windows"]) {
      const job = (/** @type {string} */ name) => graph.split("\n  " + name + ":\n")[1]?.split(/\n {2}[a-z0-9-]+:/)[0] || "",
        pack = job(`pack-${platform}-native-application`), runtime = job(`test-${platform}-packed-runtime`);
      assert.match(pack, new RegExp(`needs: assemble-${platform}-native-application`));
      assert.match(pack, /uses: \.\/\.github\/workflows\/pack-native-application\.yml/);
      for (const name of ["executable_artifact_name", "executable_sha256", "build_handoff_artifact_name", "build_handoff_bundle_sha256", "build_handoff_sha256"]) {
        const reference = name + ": " + String.fromCharCode(36) + "{{ needs.assemble-" + platform + "-native-application.outputs." + name + " }}";
        assert.ok(pack.includes(reference), reference);
      }
      assert.match(runtime, new RegExp(`needs: pack-${platform}-native-application`));
      assert.match(runtime, /uses: \.\/\.github\/workflows\/test-native-release\.yml/); assert.match(runtime, /variant: packed/);
      assert.doesNotMatch(pack, /needs:.*test-/); assert.doesNotMatch(runtime, /variant: raw/);
      const workflowOutputs = graph.slice(0, graph.indexOf("\npermissions:"));
      for (const output of ["executable_artifact_name", "executable_sha256"]) {
        assert.ok(workflowOutputs.includes("value: " + String.fromCharCode(36) + "{{ jobs.pack-" + platform + "-native-application.outputs." + output + " }}"));
      }
      assert.ok(workflowOutputs.includes("value: " + String.fromCharCode(36) + "{{ jobs.pack-" + platform + "-native-application.outputs.build_bundle_sha256 }}"));
      const rawOutputs = {
        [`${platform}_raw_executable_artifact_name`]: "executable_artifact_name",
        [`${platform}_raw_executable_sha256`]: "executable_sha256",
        [`${platform}_application_build_artifact_name`]: "build_handoff_artifact_name",
        [`${platform}_application_build_bundle_sha256`]: "build_handoff_bundle_sha256",
        [`${platform}_application_handoff_sha256`]: "build_handoff_sha256",
      };
      for (const [output, assembleOutput] of Object.entries(rawOutputs)) {
        const outputBlock = workflowOutputs.split(`\n      ${output}:\n`)[1]?.split(/\n\s{6}[a-z_]+:/)[0] || "",
          forwarding = `value: ${String.fromCharCode(36)}{{ jobs.assemble-${platform}-native-application.outputs.${assembleOutput} }}`;
        assert.ok(outputBlock.includes(forwarding), `${output} forwards the assemble job's ${assembleOutput}`);
      }
    }
  });
  test("every executable job reachable from the native graph has a one-hour ceiling", () => {
    const directory = path.join(__dirname, "../.github/workflows"), visited = new Set(); let executableJobs = 0;
    const visit = (/** @type {string} */ name) => {
      assert.match(name, /^[a-z0-9-]+\.yml$/);
      if (visited.has(name)) {return;}
      visited.add(name);
      const source = fs.readFileSync(path.join(directory, name), "utf8"), jobs = source.split(/^jobs:\s*$/m)[1];
      assert.ok(jobs, name + " declares jobs");
      for (const match of jobs.matchAll(/^ {2}([a-z0-9-]+):\n([\s\S]*?)(?=^ {2}[a-z0-9-]+:|$(?![\s\S]))/gm)) {
        const body = match[2] || "", callee = body.match(/^ {4}uses: \.\/\.github\/workflows\/([a-z0-9-]+\.yml)$/m)?.[1];
        if (callee) {visit(callee); continue;}
        const timeout = body.match(/^ {4}timeout-minutes: (.+)$/m)?.[1] || "",
          branches = /^\$\{\{ ((?:inputs\.(?:stage|phase) == '[a-z0-9-]+' && [0-9]+ \|\| )+[0-9]+) \}\}$/.exec(timeout)?.[1],
          minutes = branches ? [...branches.matchAll(/&& ([0-9]+) \|\| /g)].map(row => Number(row[1]))
            .concat(Number(branches.match(/ ([0-9]+)$/)?.[1])) : [Number(timeout)];
        // Accept only literal stage comparisons and enforce the ceiling on every outcome.
        assert.ok(minutes.every(value => Number.isInteger(value) && value >= 1 && value <= 60),
          `${name}/${match[1]} has a bounded job timeout`);
        executableJobs++;
      }
    };
    visit("build-native-prerequisites.yml");
    assert.ok(visited.size >= 9 && executableJobs > 30, "the complete native compiler/build/packing/test graph was inspected");
  });
  for (const target of ["linux", "win32"]) {
    test(`${target} uses the existing packing API with the original independent pins`, t => {
      const f = fixture(t, /** @type {'linux'|'win32'} */ (target)); f.execute(0); f.execute(1);
      const packed = f.calls.find(row => row.stage === "pack"); assert.equal(packed.options.target, target);
      assert.equal(packed.options.sourceSha256, f.env.NATIVE_RAW_SHA256);
      assert.match(fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), /packed-executable-123456789-1/);
      assert.deepEqual(f.calls.map(row => row.stage), ["restore", "curl", "tar", "pack", "seal"]);
    });
  }
  test("the shared packer reuses admitted bytes without another download or extraction", t => {
    const f = fixture(t); f.execute(0); f.execute(1);
    const count = f.calls.length, first = f.preparePacking();
    assert.equal(f.calls.length, count);
    assert.equal(first.upx, path.join(f.temp, "native-packing-tool/upx"));
  });
  test("changed cached packer bytes are rejected without replacement", t => {
    const f = fixture(t); f.execute(0); f.execute(1);
    const upx = path.join(f.temp, "native-packing-tool/upx"), count = f.calls.length;
    fs.appendFileSync(upx, "changed");
    assert.throws(() => f.preparePacking(), /Pinned packing tool bytes differ/);
    assert.equal(f.calls.length, count); assert.match(fs.readFileSync(upx, "utf8"), /changed$/);
  });
  test("partial packer caches are preserved and cannot be promoted by a retry", t => {
    const f = fixture(t); f.execute(0); f.execute(1);
    fs.unlinkSync(path.join(f.temp, "native-packing-tool/upx"));
    const count = f.calls.length;
    assert.throws(() => f.preparePacking(), /unknown or incomplete/);
    assert.equal(f.calls.length, count);
    assert.equal(fs.existsSync(path.join(f.temp, "native-packing-tool/upx.tar.xz")), true);
  });
  test("rejects a windows alias before downloading artifacts selected by canonical win32", t => {
    const f = fixture(t, "win32"); Reflect.set(f.env, "NATIVE_PLATFORM", "windows");
    assert.throws(() => f.execute(0), /exact raw executable/);
    assert.equal(f.calls.length, 0);
  });
  for (const [name, key, value] of [["another attempt", "GITHUB_RUN_ATTEMPT", "2"], ["raw alias", "NATIVE_RAW_ARTIFACT", "other"],
    ["metadata traversal", "NATIVE_METADATA_ARTIFACT", "../handoff.tar"], ["missing SHA", "NATIVE_RAW_SHA256", ""], ["invalid platform", "NATIVE_PLATFORM", "darwin"]]) {
    test(`rejects ${name} before artifact access`, t => {const f = fixture(t); assert.ok(key && value !== undefined); Reflect.set(f.env, key, value); assert.throws(() => f.execute(0), /exact raw executable/); assert.equal(f.calls.length, 0);});
  }
  test("tampered raw executable fails before restore or tool acquisition", t => {
    const f = fixture(t); fs.appendFileSync(f.raw, "changed"); assert.throws(() => f.execute(1), /regular-file pin/); assert.equal(f.calls.length, 0);
  });
  test("extra metadata member fails before extraction", t => {
    const f = fixture(t); f.members(["unexpected"]); assert.throws(() => f.execute(1), /unexpected members/); assert.equal(f.calls.length, 0);
  });
  test("a handoff for another target cannot select the packing platform", t => {
    const f = fixture(t); f.handoff.target = "x86_64-pc-windows-msvc"; f.reseal(); assert.throws(() => f.execute(1), /does not bind/);
    assert.deepEqual(f.calls.map(row => row.stage), ["restore"]);
  });
  test("changed producer bytes fail before tool acquisition", t => {
    const f = fixture(t); f.save("result.json", {changed: true}); assert.throws(() => f.execute(1), /regular-file pin/);
    assert.deepEqual(f.calls.map(row => row.stage), ["restore"]);
  });
});
