"use strict";
const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {prepareLocalWslNode} = require("../scripts/build-local-native-wsl-node"), {dependencyPins} = require("../scripts/check-native-notices");

/** Inert npm/binary fixtures only; no installation, compiler, WSL or network is started.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-wsl-node-")), workspace = path.join(base, "workspace"), npmCli = path.join(base, "npm-cli.js"),
    calls = /** @type {string[]} */ ([]), version = require("../package.json").devDependencies.esbuild;
  fs.writeFileSync(npmCli, "fixture npm\n");
  const dependencies = {npmCli, nodeVersion: dependencyPins().buildNode.version, verifyScope() {calls.push("scope");},
    run: /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ (
      (/** @type {string} */ command, /** @type {string[]} */ args) => {
        if (command === process.execPath) {
          calls.push("npm"); assert.equal(args[0], npmCli); assert.ok(args.includes("--ignore-scripts")); assert.ok(args.includes("--include=optional"));
          const userConfig = args.find(arg => arg.startsWith("--userconfig="))?.slice("--userconfig=".length),
            globalConfig = args.find(arg => arg.startsWith("--globalconfig="))?.slice("--globalconfig=".length);
          assert.ok(userConfig && globalConfig); assert.notEqual(userConfig, globalConfig);
          assert.equal(fs.readFileSync(userConfig, "utf8"), ""); assert.equal(fs.readFileSync(globalConfig, "utf8"), "");
          const root = args[args.indexOf("--prefix") + 1]; assert.ok(root);
          const binary = path.join(root, "node_modules/@esbuild/linux-x64/bin/esbuild");
          fs.mkdirSync(path.dirname(binary), {recursive: true}); fs.writeFileSync(binary, "fixture Linux esbuild\n"); fs.chmodSync(binary, 0o755);
          fs.mkdirSync(path.join(root, "npm-cache")); fs.writeFileSync(path.join(root, "npm-cache/temporary"), "reproducible npm cache\n");
          return {status: 0, signal: null};
        }
        calls.push("version"); assert.match(command, /\/node_modules\/@esbuild\/linux-x64\/bin\/esbuild$/);
        return {status: 0, signal: null, stdout: version + "\n"};
      }))};
  t.after(() => fs.rmSync(base, {recursive: true, force: true}));
  return {base, workspace, calls, dependencies};
}

describe("Separate WSL build-only Node dependencies", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("uses original lockfiles and reuses the Linux bundler without npm or writes", t => {
    const f = fixture(t), first = prepareLocalWslNode(f.workspace, f.dependencies), receipt = path.join(first.root, "admitted.json"),
      before = fs.statSync(receipt).mtimeMs, count = f.calls.length, second = prepareLocalWslNode(f.workspace, f.dependencies);
    assert.equal(first.reused, false); assert.equal(second.reused, true); assert.equal(first.sha256, second.sha256);
    assert.deepEqual(f.calls.slice(count), ["scope"]); assert.equal(fs.statSync(receipt).mtimeMs, before);
    assert.equal(fs.existsSync(path.join(first.root, "npm-cache")), false);
    assert.deepEqual(fs.readFileSync(path.join(first.root, "package-lock.json")), fs.readFileSync(path.join(__dirname, "../package-lock.json")));
  });
  test("rejects changed binaries and unknown generation state instead of reinstalling", t => {
    const f = fixture(t), result = prepareLocalWslNode(f.workspace, f.dependencies); fs.appendFileSync(result.binary, "changed");
    assert.throws(() => prepareLocalWslNode(f.workspace, f.dependencies), /dependency or original lock/);
    assert.equal(f.calls.filter(call => call === "npm").length, 1);
    const g = fixture(t), fresh = prepareLocalWslNode(g.workspace, g.dependencies); fs.writeFileSync(path.join(fresh.root, "unknown"), "keep");
    assert.throws(() => prepareLocalWslNode(g.workspace, g.dependencies), /unknown state/);
    assert.equal(fs.readFileSync(path.join(fresh.root, "unknown"), "utf8"), "keep");
  });
  test("preserves failed npm output and refuses an implicit retry", t => {
    const f = fixture(t), deps = {...f.dependencies, run: /** @type {typeof f.dependencies.run} */ (/** @type {unknown} */ (() => ({status: 1})))};
    assert.throws(() => prepareLocalWslNode(f.workspace, deps), /installation failed; preserve/);
    assert.throws(() => prepareLocalWslNode(f.workspace, f.dependencies), /incomplete/);
    assert.equal(f.calls.includes("npm"), false);
  });
  test("requires the enforced scope and pinned Node before creating a workspace", t => {
    const f = fixture(t);
    assert.throws(() => prepareLocalWslNode(f.workspace, {...f.dependencies, verifyScope() {throw new Error("scope missing");}}), /scope missing/);
    assert.throws(() => prepareLocalWslNode(f.workspace, {...f.dependencies, nodeVersion: "v22.0.0"}), /pinned build Node/);
    assert.equal(fs.existsSync(f.workspace), false);
  });
});
