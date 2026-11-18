"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"),
  path = require("node:path"), os = require("node:os");
const {fileHash} = require("../scripts/build-source-inputs");
const {publishLocalDistribution, launchLocalNativeRelease, runLocalNativeRelease} = require("../scripts/build-local-native-command");

/** Only inert archive bytes; no compiler, packer or miner execution.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-publish-")), source = path.join(root, "source"),
    destination = path.join(root, "destination"), name = "mom-v0.9.4-lin.tgz";
  fs.mkdirSync(source); fs.mkdirSync(destination);
  const archive = path.join(source, name), sidecar = archive + ".sha256";
  fs.writeFileSync(archive, "new archive"); const sha256 = fileHash(archive);
  fs.writeFileSync(sidecar, `${sha256}  ${name}\n`);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, source, destination, name, result: {archive: {path: archive, sha256}, sidecar},
    archive: path.join(destination, name), sidecar: path.join(destination, name + ".sha256")};
}

/** @param {ReturnType<typeof fixture>} f */
function oldPair(f) {fs.writeFileSync(f.archive, "old archive"); fs.writeFileSync(f.sidecar, "old checksum");}

test.describe("Local release host routing", () => {
  test("routes Windows x64 through the WSL coordinator after version and environment admission", () => {
    const spawn = () => {throw new Error("unexpected direct process launch");};
    const status = launchLocalNativeRelease({CXXFLAGS: "-march=native", BASH_ENV: "/untrusted/hook"}, spawn, {
      platform: "win32", arch: "x64", launchWindows(environment, suppliedSpawn) {
        assert.equal(suppliedSpawn, spawn); assert.equal(environment["CXXFLAGS"], undefined);
        assert.equal(environment["BASH_ENV"], undefined); assert.equal(environment["MOM_PORTABLE_BUILD"], "1"); return 7;
      },
    });
    assert.equal(status, 7);
    assert.throws(() => launchLocalNativeRelease({MOM_RELEASE_VERSION: "../bad"}, spawn, {
      platform: "win32", arch: "x64", launchWindows() {assert.fail("invalid version reached Windows");},
    }), /version/i);
  });
  test("rejects unsupported hosts without invoking either builder", () => {
    assert.throws(() => launchLocalNativeRelease({}, () => {assert.fail("process launched");}, {
      platform: "darwin", arch: "arm64", launchWindows() {assert.fail("Windows builder invoked");},
    }), /Linux x64 or Windows x64/);
  });
});

test.describe("Local release command containment", {skip: process.platform !== "linux" || process.arch !== "x64", concurrency: false}, () => {
  /** No installed SDK, systemd scope, source fetch or build is used by these fixtures.
   * @param {import('node:test').TestContext} t */
  function tools(t) {
    t.mock.method(require("../scripts/build-local-native-linux-sdk"), "preflightLocalLinuxSdk", () => ({}));
    t.mock.method(require("../scripts/build-local-native-sdk"), "localTools", () => ({systemdRun: "/usr/bin/systemd-run"}));
    t.mock.method(require("../scripts/build-local-native-sdk"), "defaultWorkspace", () => "/fixture/sdk");
  }
  test("locks before launching a host-memory-capped complete build", t => {
    tools(t);
    const code = launchLocalNativeRelease({PATH: "/usr/bin", BASH_ENV: "/untrusted/hook", CXXFLAGS: "-march=native",
      "BASH_FUNC_flock%%": "() { return 0; }"}, (command, args, options) => {
      assert.equal(command, "bash");
      assert.match(args[2] || "", /flock 9[\s\S]*exec "\$@"/);
      assert.ok(args.includes("--property=MemoryMax=4G"));
      assert.ok(args.includes("--property=MemorySwapMax=0"));
      assert.deepEqual(args.slice(-2), ["--build", "/fixture/sdk"]);
      assert.equal(options.env?.["BASH_ENV"], undefined);
      assert.equal(options.env?.["BASH_FUNC_flock%%"], undefined);
      assert.equal(options.env?.["CXXFLAGS"], undefined);
      assert.equal(options.env?.["MOM_BUILD_JOBS"], "2");
      assert.equal(options.stdio, "inherit");
      return {status: 0};
    });
    assert.equal(code, 0);
  });
  test("prerequisite failure never launches a build or creates its workspace", t => {
    tools(t);
    t.mock.method(require("../scripts/build-local-native-linux-sdk"), "preflightLocalLinuxSdk", () => {throw new Error("missing pinned prerequisite");});
    t.mock.method(require("../scripts/build-local-native-sdk"), "defaultWorkspace", () => {assert.fail("workspace touched");});
    assert.throws(() => launchLocalNativeRelease({}, () => {assert.fail("build launched");}), /missing pinned prerequisite/);
  });
  test("invalid release versions fail before SDK preparation or cache mutation", t => {
    tools(t);
    t.mock.method(require("../scripts/build-local-native-linux-sdk"), "preflightLocalLinuxSdk", () => {assert.fail("SDK preflight reached");});
    t.mock.method(require("../scripts/build-local-native-sdk"), "defaultWorkspace", () => {assert.fail("cache touched");});
    assert.throws(() => launchLocalNativeRelease({MOM_RELEASE_VERSION: "../bad"}, () => {assert.fail("build launched");}), /version/i);
  });
  test("nested Engineering builds do not acquire the parent's build lock again", t => {
    tools(t);
    for (const marker of ["MOM_R_SH", "MOM_NATIVE_BUILD_LOCK"]) {
      assert.equal(launchLocalNativeRelease({[marker]: "1"}, (_command, args) => {
        // Empty PATH ensures an attempted git/mkdir/flock acquisition would fail.
        return require("node:child_process").spawnSync("/bin/bash", ["-eu", "-c", args[2] || "", "--", process.execPath, "-e", ""], {
          env: {[marker]: "1", PATH: ""}, timeout: 2000});
      }), 0);
    }
  });
  test("actual no-argument npm entrypoint selects native preparation and preserves its exit status", t => {
    const f = fixture(t), scripts = path.join(f.root, "scripts"), metadata = require("../package.json");
    fs.mkdirSync(scripts);
    fs.writeFileSync(path.join(f.root, "package.json"), JSON.stringify({version: metadata.version,
      scripts: {"build:release": metadata.scripts["build:release"]}}));
    for (const name of ["build-release.js", "build-environment.js", "build-source-inputs.js"]) {
      fs.copyFileSync(path.join(__dirname, "../scripts", name), path.join(scripts, name));
    }
    fs.writeFileSync(path.join(scripts, "build-local-native-command.js"),
      'const release=require("./build-release");if(typeof release.releaseVersion!=="function")throw Error("Release APIs unavailable during CLI dispatch");\n' +
      'exports.launchLocalNativeRelease=()=>{console.log("FIXTURE_NATIVE_PREPARATION");return Number(process.env.FIXTURE_STATUS);};\n');
    /** @type {NodeJS.ProcessEnv} */
    const env = {...process.env, npm_config_update_notifier: "false", npm_config_audit: "false"};
    for (const key of Object.keys(env)) {if (key.startsWith("MOM_NATIVE_") || key === "NODE_OPTIONS") {delete env[key];}}
    for (const status of [0, 7]) {
      const result = require("node:child_process").spawnSync("npm", ["run", "build:release"], {
        cwd: f.root, env: {...env, FIXTURE_STATUS: String(status)}, encoding: "utf8", timeout: 20000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, status, result.stderr);
      assert.equal(result.stdout.split("FIXTURE_NATIVE_PREPARATION").length, 2);
    }
  });
  test("propagates scope/build failure instead of publishing success", t => {
    tools(t); assert.equal(launchLocalNativeRelease({}, () => ({status: 7})), 7);
    assert.equal(launchLocalNativeRelease({}, () => ({status: null})), 1);
    assert.throws(() => launchLocalNativeRelease({}, () => ({status: null, error: new Error("scope failed")})), /scope failed/);
  });
  test("preserves one failed application output and refuses to grow it on retry", t => {
    const f = fixture(t), workspace = path.join(f.root, "sdk"), local = require("../scripts/build-local-native-sdk"),
      stage = path.join(workspace, "release-work");
    t.mock.method(local, "verifyCurrentMemoryScope", () => undefined);
    t.mock.method(local, "ensureLocalLinuxSysroot", (/** @type {string} */ _workspace, /** @type {NodeJS.ProcessEnv} */ environment) => environment);
    const sdk = t.mock.method(require("../scripts/build-local-native-linux-sdk"), "buildLocalLinuxSdk", () => ({}));
    t.mock.method(require("../scripts/build-local-native-distribution"), "buildLocalNativeDistribution", () => {
      fs.writeFileSync(path.join(stage, "diagnostic.log"), "preserve"); throw new Error("fixture link failure");
    });
    t.mock.method(console, "error", () => undefined);
    assert.throws(() => runLocalNativeRelease(workspace), /fixture link failure/);
    assert.throws(() => runLocalNativeRelease(workspace), /Inspect preserved release work/);
    assert.equal(sdk.mock.callCount(), 1);
    assert.equal(fs.readFileSync(path.join(stage, "diagnostic.log"), "utf8"), "preserve");
  });
  test("a prerequisite failure leaves no empty application work slot", t => {
    const f = fixture(t), workspace = path.join(f.root, "sdk");
    t.mock.method(require("../scripts/build-local-native-sdk"), "verifyCurrentMemoryScope", () => undefined);
    t.mock.method(require("../scripts/build-local-native-sdk"), "ensureLocalLinuxSysroot",
      (/** @type {string} */ _workspace, /** @type {NodeJS.ProcessEnv} */ environment) => environment);
    t.mock.method(require("../scripts/build-local-native-linux-sdk"), "buildLocalLinuxSdk", () => {throw new Error("fixture SDK failure");});
    assert.throws(() => runLocalNativeRelease(workspace), /fixture SDK failure/);
    assert.equal(fs.existsSync(path.join(workspace, "release-work")), false);
  });
});

test.describe("Local native archive publication", {concurrency: false}, () => {
  test("publishes the exact verified bytes and removes private staging", t => {
    const f = fixture(t), result = publishLocalDistribution(f.result, f.destination);
    assert.equal(fileHash(result.archive), f.result.archive.sha256);
    assert.equal(fs.readFileSync(result.sidecar, "utf8"), `${result.sha256}  ${f.name}\n`);
    assert.deepEqual(fs.readdirSync(f.destination).sort(), [f.name, f.name + ".sha256"].sort());
  });
  test("replaces only the regular output pair after both new files are staged", t => {
    const f = fixture(t); oldPair(f); publishLocalDistribution(f.result, f.destination);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "new archive");
    assert.equal(fs.readFileSync(f.sidecar, "utf8"), fs.readFileSync(f.result.sidecar, "utf8"));
  });
  test("rejects a changed archive before touching existing outputs", t => {
    const f = fixture(t); oldPair(f); fs.appendFileSync(f.result.archive.path, "changed");
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /verified archive/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "old archive");
  });
  test("rejects malformed checksum text before publication", t => {
    const f = fixture(t); fs.writeFileSync(f.result.sidecar, "wrong");
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /exact checksum/);
    assert.deepEqual(fs.readdirSync(f.destination), []);
  });
  test("rejects checksum bytes changed during staging without withdrawing prior outputs", t => {
    const f = fixture(t); oldPair(f); const copy = fs.copyFileSync;
    t.mock.method(fs, "copyFileSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to, /** @type {number|undefined} */ flags) => {
      copy(from, to, flags); if (from === f.result.sidecar) {fs.writeFileSync(to, "changed");}
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /changed while copying/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "old archive");
    assert.equal(fs.readFileSync(f.sidecar, "utf8"), "old checksum");
  });
  test("exclusive installation does not overwrite a concurrently created foreign archive", t => {
    const f = fixture(t), link = fs.linkSync;
    t.mock.method(fs, "linkSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (to === f.archive) {fs.writeFileSync(to, "foreign");} return link(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /EEXIST/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "foreign");
    assert.equal(fs.existsSync(f.sidecar), false);
  });
  test("a replacement during withdrawal is restored rather than discarded", t => {
    const f = fixture(t); oldPair(f); const rename = fs.renameSync, original = path.join(f.root, "original-archive");
    t.mock.method(fs, "renameSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (from === f.archive) {rename(from, original); fs.writeFileSync(from, "foreign");}
      return rename(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /changed while withdrawing/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "foreign");
    assert.equal(fs.readFileSync(original, "utf8"), "old archive");
    assert.equal(fs.readFileSync(f.sidecar, "utf8"), "old checksum");
  });
  test("refuses directory outputs without replacing their contents", t => {
    const f = fixture(t); fs.mkdirSync(f.archive); fs.writeFileSync(path.join(f.archive, "keep"), "keep");
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /non-regular release output/);
    assert.equal(fs.readFileSync(path.join(f.archive, "keep"), "utf8"), "keep");
  });
  test("refuses a symlink output without changing its target", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), target = path.join(f.root, "keep"); fs.writeFileSync(target, "keep"); fs.symlinkSync(target, f.archive);
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /non-regular release output/);
    assert.equal(fs.readFileSync(target, "utf8"), "keep");
  });
  test("refuses a symlink input even when its bytes match", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), alias = path.join(f.source, "alias.sha256"); fs.symlinkSync(f.result.sidecar, alias);
    assert.throws(() => publishLocalDistribution({...f.result, sidecar: alias}, f.destination), /canonical regular file/);
  });
  test("restores the old pair when installation fails after replacing the archive", t => {
    const f = fixture(t); oldPair(f); const link = fs.linkSync;
    t.mock.method(fs, "linkSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (String(from).endsWith(path.sep + f.name + ".sha256") && to === f.sidecar) {throw new Error("fixture install failure");}
      return link(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /fixture install failure/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "old archive");
    assert.equal(fs.readFileSync(f.sidecar, "utf8"), "old checksum");
    assert.equal(fs.readdirSync(f.destination).length, 2);
  });
  test("a failed fresh install leaves no checksum or partial archive", t => {
    const f = fixture(t), link = fs.linkSync;
    t.mock.method(fs, "linkSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (to === f.sidecar) {throw new Error("fixture install failure");} return link(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /fixture install failure/);
    assert.deepEqual(fs.readdirSync(f.destination), []);
  });
  test("failed recovery preserves original files in the reported private stage", t => {
    const f = fixture(t); oldPair(f); const link = fs.linkSync;
    t.mock.method(fs, "linkSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (String(from).endsWith(path.sep + f.name + ".sha256") && to === f.sidecar || String(from).includes(path.sep + "old-")) {
        throw new Error("fixture install/recovery failure");
      }
      return link(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /recovery failed; preserved originals/);
    const stages = fs.readdirSync(f.destination).filter(name => name.startsWith("mom-release-publish-")); assert.equal(stages.length, 1);
    assert.equal(fs.readFileSync(path.join(f.destination, stages[0] || "", "old-" + f.name), "utf8"), "old archive");
  });
  test("recovery never deletes a foreign replacement of an installed archive", t => {
    const f = fixture(t); oldPair(f); const link = fs.linkSync;
    t.mock.method(fs, "linkSync", (/** @type {fs.PathLike} */ from, /** @type {fs.PathLike} */ to) => {
      if (to === f.sidecar) {
        fs.unlinkSync(f.archive); fs.writeFileSync(f.archive, "foreign"); throw new Error("fixture race");
      }
      return link(from, to);
    });
    assert.throws(() => publishLocalDistribution(f.result, f.destination), /recovery failed/);
    assert.equal(fs.readFileSync(f.archive, "utf8"), "foreign");
    const stage = fs.readdirSync(f.destination).find(name => name.startsWith("mom-release-publish-")); assert.ok(stage);
    assert.equal(fs.readFileSync(path.join(f.destination, stage, "old-" + f.name), "utf8"), "old archive");
  });
});
