"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), {spawnSync} = require("node:child_process");
const {test, describe} = require("node:test"), assert = require("node:assert/strict");
const {prepareSdkSources, verifyPreparedSdkSources, validateSourceRoots, sdkSourceFiles, sdkSourceEnvironment, sdkPatchScope} = require("../scripts/prepare-native-sdk-sources");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs"), {syclRevision} = require("../scripts/probe-static-sycl");
const repo = path.resolve(__dirname, ".."), patches = ["runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"];

/** Only small local files/Git fixtures; no SDK, network, compiler, GPU or remote host. @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-acpp-sources-")), source = path.join(root, "source"), output = path.join(root, "output");
  fs.mkdirSync(source); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, source, output, roots: {acpp: source}};
}

/** A deliberately incomplete checkout must never become valid just by matching a JSON hash.
 * @param {ReturnType<typeof fixture>} f */
function fakeReceipt(f) {
  fs.writeFileSync(path.join(f.source, "header.h"), "fixture header\n");
  const files = sdkSourceFiles(f.source), receipt = {version: 2, target: "acpp", status: "PASS", compiled: false, buildOnly: true, publicationQualified: false,
    recipe: sourceInputs(repo, [], ["scripts/prepare-native-sdk-sources.js", "scripts/probe-static-sycl.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-hcf.js", ...patches]),
    prepared: {acpp: {root: f.source, revision: syclRevision, paths: [], leaves: [], files,
      patches: Object.fromEntries(patches.map(file => [path.join(repo, file), fileHash(path.join(repo, file))]))}}};
  const file = path.join(f.root, "sources.json"); fs.writeFileSync(file, JSON.stringify(receipt)); return file;
}

describe("Pinned AdaptiveCpp offline source preparation", () => {
  test("requires explicit build-only admission before inspecting roots or creating output", t => {
    const f = fixture(t); assert.throws(() => prepareSdkSources("acpp", f.roots, f.output, {}), /MOM_AOT_BUILD_ONLY/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("accepts exactly one source root, not a runtime or compiler cohort", t => {
    const f = fixture(t); assert.deepEqual(validateSourceRoots(f.roots, f.output, {acpp: true}), f.roots);
    for (const roots of [{}, {acpp: f.source, extra: f.source}, {dpcpp: f.source}, {acpp: null}]) {
      assert.throws(() => validateSourceRoots(roots, f.output, {acpp: true}));
    }
  });
  test("does not overwrite an existing output or source checkout", t => {
    const f = fixture(t); fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "preserve");
    assert.throws(() => prepareSdkSources("acpp", f.roots, f.output, {MOM_AOT_BUILD_ONLY: "1"}), /new absolute output/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "preserve");
    assert.throws(() => validateSourceRoots(f.roots, path.join(f.source, "child"), {acpp: true}), /overlaps/);
  });
  test("source aliases cannot hide output/source overlap", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), alias = path.join(f.root, "alias"); fs.symlinkSync(f.source, alias, "dir");
    assert.throws(() => validateSourceRoots({acpp: alias}, f.output, {acpp: true}), /real directory/);
    assert.throws(() => validateSourceRoots(f.roots, path.join(alias, "child"), {acpp: true}), /overlaps/);
  });
  test("missing original pinned objects fail before producing a checkout", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), init = spawnSync("git", ["init", "--quiet", "--template=", f.source], {encoding: "utf8", env: sdkSourceEnvironment(process.env)});
    assert.equal(init.status, 0, init.stderr);
    assert.throws(() => prepareSdkSources("acpp", f.roots, f.output, {...process.env, MOM_AOT_BUILD_ONLY: "1"}), /source input query failed: acpp/);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("only the Git-aware inventory excludes its own root metadata", t => {
    const f = fixture(t); fs.mkdirSync(path.join(f.source, ".git")); fs.writeFileSync(path.join(f.source, ".git/HEAD"), "metadata\n");
    fs.writeFileSync(path.join(f.source, "header.h"), "tracked\n");
    assert.deepEqual(Object.keys(sdkSourceFiles(f.source, true)), ["header.h"]);
    assert.deepEqual(Object.keys(sdkSourceFiles(f.source)), [".git/HEAD", "header.h"], "archive cohorts keep complete membership checks");
  });
  test("Git-aware inventory still rejects unexpected source symlinks", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.symlinkSync(f.root, path.join(f.source, "escape"), "dir");
    assert.throws(() => sdkSourceFiles(f.source, true), /bounded regular files/);
  });
  test("AdaptiveCpp applies both complete reviewed patches, not runtime-only SYCL hunks", () => {
    for (const patch of patches) {assert.deepEqual(sdkPatchScope("acpp", path.join(repo, patch)), ["--index"]);}
  });
  test("new patch files remain visible to exact Git-diff admission", t => {
    const f = fixture(t), env = sdkSourceEnvironment(process.env);
    const git = (/** @type {string[]} */ args) => {
      const result = spawnSync("git", ["-C", f.source, ...args], {env, encoding: "utf8"});
      assert.equal(result.status, 0, result.stderr); return result.stdout;
    };
    git(["init", "--quiet", "--template="]);
    fs.writeFileSync(path.join(f.source, "original.h"), "original\n");
    fs.writeFileSync(path.join(f.source, "unchanged.h"), "unchanged\n"); git(["add", "--", "original.h", "unchanged.h"]);
    git(["-c", "user.name=Source fixture", "-c", "user.email=fixture@invalid", "commit", "--quiet", "-m", "Original source"]);
    const patch = path.join(f.root, "reviewed.patch"), scope = sdkPatchScope("acpp", patch);
    fs.writeFileSync(patch, "diff --git a/original.h b/original.h\n--- a/original.h\n+++ b/original.h\n@@ -1 +1 @@\n-original\n+reviewed edit\ndiff --git a/added.h b/added.h\nnew file mode 100644\n--- /dev/null\n+++ b/added.h\n@@ -0,0 +1 @@\n+reviewed header\n");
    git(["apply", ...scope, "--check", "--", patch]); git(["apply", ...scope, "--", patch]);
    assert.match(git(["diff", "HEAD", "--"]), /new file mode 100644[\s\S]*reviewed header/);
    assert.equal(git(["ls-files", "--others", "--exclude-standard"]), "", "patch additions must not become invisible untracked files");
    assert.equal(git(["ls-files"]).trim(), "added.h\noriginal.h\nunchanged.h", "patch application must preserve unchanged tracked inputs");
  });
  test("wrong-target and incomplete JSON cannot certify a prepared checkout", t => {
    const f = fixture(t), file = path.join(f.root, "invalid.json");
    for (const target of ["linux", "windows", "compiler", "acpp"]) {
      fs.writeFileSync(file, JSON.stringify({version: 2, target, status: "PASS"}));
      assert.throws(() => verifyPreparedSdkSources(file, "acpp"), /wrong-target/);
    }
  });
  test("a Git metadata symlink cannot turn a recorded inventory into a source proof", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), file = fakeReceipt(f), foreign = path.join(f.root, "foreign"); fs.mkdirSync(foreign);
    fs.symlinkSync(foreign, path.join(f.source, ".git"), "dir");
    assert.throws(() => verifyPreparedSdkSources(file, "acpp"), /independent regular Git directory/);
    assert.deepEqual(fs.readdirSync(foreign), []);
  });
  test("borrowed Git objects are rejected without following or rewriting them", t => {
    const f = fixture(t), file = fakeReceipt(f); fs.mkdirSync(path.join(f.source, ".git/objects/info"), {recursive: true});
    const alternates = path.join(f.source, ".git/objects/info/alternates"); fs.writeFileSync(alternates, "/unrelated/objects\n");
    assert.throws(() => verifyPreparedSdkSources(file, "acpp"), /independent regular Git directory/);
    assert.equal(fs.readFileSync(alternates, "utf8"), "/unrelated/objects\n");
  });
  test("Git and extraction environment sanitization does not mutate the caller", () => {
    const input = {PATH: "/usr/bin", GIT_DIR: "/wrong", GIT_CONFIG_COUNT: "9", GIT_ALLOW_PROTOCOL: "https", TAR_OPTIONS: "--to-command=wrong"}, original = {...input};
    const safe = sdkSourceEnvironment(input); assert.equal(safe["GIT_DIR"], undefined); assert.equal(safe["GIT_CONFIG_COUNT"], undefined);
    assert.equal(safe["GIT_ALLOW_PROTOCOL"], undefined); assert.equal(safe["TAR_OPTIONS"], undefined); assert.equal(safe.GIT_NO_LAZY_FETCH, "1"); assert.deepEqual(input, original);
  });
});
