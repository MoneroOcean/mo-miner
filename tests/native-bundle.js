"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {nativePathSource, nativeApplicationRoot, controllerBundleOptions, controllerAotCommand, verifyBundle} = require("../scripts/native-bundle");

test.describe("Native bundle reproducibility and immutable parity inputs", () => {
  const repo = path.resolve(__dirname, "..");
  let root = "", serial = 0;
  test.before(() => {root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-bundle-"));});
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  function fixture() {
    const directory = path.join(root, String(++serial)); fs.mkdirSync(directory);
    const entry = path.join(directory, "entry.js"), bundle = path.join(directory, "controller.js");
    fs.writeFileSync(entry, `module.exports = require(${JSON.stringify(path.join(repo, "compiler-policy.js"))});\n`);
    const options = controllerBundleOptions(entry, bundle);
    return {directory, entry, bundle, options};
  }
  test("bundle keeps canonical roots and function names without syntax minification", () => {
    const {entry, bundle, options} = fixture();
    assert.equal(options.absWorkingDir, repo);
    assert.equal(controllerBundleOptions(entry, bundle, true).absWorkingDir, repo);
    assert.equal(options.keepNames, true);
    assert.equal(options.minifyIdentifiers, true); assert.equal(options.minifyWhitespace, true);
    assert.notEqual(options.minify, true); assert.notEqual(options.minifySyntax, true);
  });
  test("target platform roots produce the exact stable embedded-policy path", () => {
    const linuxRoot = nativeApplicationRoot("linux"), windowsRoot = nativeApplicationRoot("win32");
    assert.equal(linuxRoot, "/__mom_builtin__");
    assert.equal(windowsRoot, "C:\\__mom_builtin__");
    assert.equal(path.posix.join(linuxRoot, "GPU-CONFIG.md"), "/__mom_builtin__/GPU-CONFIG.md");
    assert.equal(path.win32.join(windowsRoot, "GPU-CONFIG.md"), "C:\\__mom_builtin__\\GPU-CONFIG.md");
    assert.throws(() => nativeApplicationRoot("darwin"), /linux or win32/);
    const {entry, bundle} = fixture();
    assert.equal(controllerBundleOptions(entry, bundle, false, "win32").define?.["__dirname"], JSON.stringify(windowsRoot));
  });
  test("reviewed path builtin source is exactly the source embedded by controller aliases", () => {
    const {directory, options} = fixture(), pathBuiltin = nativePathSource(), aliases = options.alias || {},
      shim = path.join(repo, "runtime", "path-internals.js");
    assert.equal(fs.readFileSync(path.join(directory, "node-path.cjs"), "utf8"),
      `const {primordials} = require(${JSON.stringify(shim)});\n${pathBuiltin.source}`);
    assert.equal(aliases["node:path"], path.join(directory, "node-path.cjs"));
    for (const name of pathBuiltin.internals) {assert.equal(aliases[name], shim);}
  });
  test("read-only verification returns an in-memory bundle instead of replacing the built artifact", () => {
    const {entry, bundle, options} = fixture();
    assert.equal(options.write, true); assert.equal(controllerBundleOptions(entry, bundle, true).write, false);
  });
  test("repository and external invocation directories produce identical actual maintained-source bundles", () => {
    const {entry, bundle} = fixture();
    const script = `const esbuild=require(${JSON.stringify(require.resolve("esbuild"))});` +
      `const {controllerBundleOptions,verifyBundle}=require(${JSON.stringify(path.join(repo, "scripts/native-bundle.js"))});` +
      `const result=esbuild.buildSync(controllerBundleOptions(${JSON.stringify(entry)},${JSON.stringify(bundle)},true));` +
      "console.log(JSON.stringify(verifyBundle(result.outputFiles[0].contents,result.outputFiles[0].contents)));";
    const results = [repo, root].map((cwd) => {
      const result = spawnSync(process.execPath, ["--max-old-space-size=128", "-e", script], {cwd, encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
      return JSON.parse(result.stdout);
    });
    assert.deepEqual(results[0], results[1]); assert.ok(results[0].bytes > 10000);
  });
  test("fresh build directories produce byte-identical maintained-source bundles without temporary paths", () => {
    const esbuild = require("esbuild"), first = fixture(), second = fixture();
    const bundles = [first, second].map(({directory, options, bundle}) => {
      esbuild.buildSync(options);
      const bytes = fs.readFileSync(bundle);
      assert.equal(bytes.includes(Buffer.from(directory)), false, "Temporary build paths must not become controller data");
      assert.equal(bytes.includes(Buffer.from(repo)), false, "Checkout paths must not become controller data");
      return bytes;
    });
    verifyBundle(bundles[0] || Buffer.alloc(0), bundles[1] || Buffer.alloc(0));
  });
  test("full application bundles replace the checkout root for each target platform", () => {
    const esbuild = require("esbuild");
    for (const platform of ["linux", "win32"]) {
      const {entry, bundle} = fixture();
      fs.writeFileSync(entry, `require(${JSON.stringify(path.join(repo, "mom.js"))});\n`);
      const options = controllerBundleOptions(entry, bundle, false, platform);
      assert.equal(options.define?.["__dirname"], JSON.stringify(nativeApplicationRoot(platform)));
      esbuild.buildSync(options);
      const bytes = fs.readFileSync(bundle);
      assert.equal(bytes.includes(Buffer.from(repo)), false, "Checkout path must not become full application bundle data");
    }
  });
  test("Linux and Windows controller AOT share stable filename arguments and private working directories", () => {
    const tool = path.join(root, "sdk with spaces", "shermes"), first = fixture(), second = fixture();
    const plans = [first, second].map(({directory, bundle}) => controllerAotCommand(tool, bundle, path.join(directory, "controller.c")));
    assert.equal(plans[0]?.command, tool); assert.deepEqual(plans[0]?.args, plans[1]?.args);
    assert.deepEqual(plans[0]?.args.slice(-3), ["-o", "./controller.c", "./controller.js"]);
    assert.equal(plans[0]?.cwd, first.directory); assert.equal(plans[1]?.cwd, second.directory);
    assert.throws(() => controllerAotCommand("shermes", first.bundle, path.join(first.directory, "controller.c")), /absolute/);
    assert.throws(() => controllerAotCommand(tool, first.bundle, path.join(second.directory, "controller.c")), /colocated/);
  });
  test("real Hermes emits identical native controller data in separate fresh directories", {
    skip: !process.env["MOM_TEST_SHERMES"] && "Set MOM_TEST_SHERMES to exercise the pinned AOT compiler",
  }, () => {
    const tool = process.env["MOM_TEST_SHERMES"] || "", esbuild = require("esbuild");
    const generated = [fixture(), fixture()].map(({directory, entry, bundle, options}) => {
      fs.writeFileSync(entry, `require(${JSON.stringify(path.join(repo, "mom.js"))});\n`);
      esbuild.buildSync(options);
      const cSource = path.join(directory, "controller.c"), plan = controllerAotCommand(tool, bundle, cSource);
      const result = spawnSync(plan.command, plan.args, {cwd: plan.cwd, encoding: "utf8", timeout: 30000, maxBuffer: 8 << 20});
      assert.equal(result.error, undefined); assert.equal(result.signal, null); assert.equal(result.status, 0, result.stderr.slice(0, 4096));
      const bytes = fs.readFileSync(cSource);
      assert.equal(bytes.includes(Buffer.from(directory)), false, "AOT filenames must not expose temporary build paths");
      return bytes;
    });
    verifyBundle(generated[0] || Buffer.alloc(0), generated[1] || Buffer.alloc(0));
  });
  test("verification preserves generated source and provenance contents and modification times", () => {
    const {directory, entry, bundle} = fixture();
    const files = ["node-path.cjs", "node-path-source.json"].map((name) => path.join(directory, name));
    const before = files.map((file) => ({bytes: fs.readFileSync(file), time: fs.statSync(file, {bigint: true}).mtimeNs}));
    controllerBundleOptions(entry, bundle, true);
    files.forEach((file, index) => {
      assert.ok(fs.readFileSync(file).equals(before[index]?.bytes || Buffer.alloc(0)));
      assert.equal(fs.statSync(file, {bigint: true}).mtimeNs, before[index]?.time);
    });
  });
  test("altered generated source fails verification without repair or overwrite", () => {
    const {directory, entry, bundle} = fixture(), file = path.join(directory, "node-path.cjs");
    fs.appendFileSync(file, "\n// altered\n"); const changed = fs.readFileSync(file);
    assert.throws(() => controllerBundleOptions(entry, bundle, true), /source\/provenance differs/);
    assert.ok(fs.readFileSync(file).equals(changed));
  });
  test("a different build-Node provenance fails without replacing the receipt", () => {
    const {directory, entry, bundle} = fixture(), file = path.join(directory, "node-path-source.json");
    fs.writeFileSync(file, '{"node":"different"}\n');
    assert.throws(() => controllerBundleOptions(entry, bundle, true), /source\/provenance differs/);
    assert.equal(fs.readFileSync(file, "utf8"), '{"node":"different"}\n');
  });
  test("missing provenance cannot be silently reconstructed during qualification", () => {
    const {directory, entry, bundle} = fixture(), file = path.join(directory, "node-path-source.json");
    fs.unlinkSync(file); assert.throws(() => controllerBundleOptions(entry, bundle, true), {code: "ENOENT"});
    assert.equal(fs.existsSync(file), false);
  });
  test("relative entry or output paths fail before generating inputs", () => {
    assert.throws(() => controllerBundleOptions("entry.js", path.join(root, "bundle.js")), /absolute/);
    assert.throws(() => controllerBundleOptions(path.join(root, "entry.js"), "bundle.js"), /absolute/);
  });
});

test.describe("Native bundle bounded mismatch diagnostics", () => {
  test("equal bytes return only a bounded hash and size receipt", () => {
    const bytes = Buffer.from("same source"), result = verifyBundle(bytes, Uint8Array.from(bytes));
    assert.equal(result.bytes, bytes.length); assert.match(result.sha256, /^[0-9a-f]{64}$/);
  });
  test("different megabyte buffers fail with small hash diagnostics, never an assertion diff", () => {
    assert.throws(() => verifyBundle(Buffer.alloc(1 << 20, 0x61), Buffer.alloc(1 << 20, 0x62)), (error) => {
      assert.ok(error instanceof Error); assert.equal(error.name, "Error"); assert.ok(error.message.length < 256);
      assert.match(error.message, /^Native bundle mismatch: actual 1048576 bytes SHA256 [a-f0-9]{64}; expected 1048576 bytes SHA256 [a-f0-9]{64}$/);
      return true;
    });
  });
  test("different lengths also fail without rendering either input", () => {
    assert.throws(() => verifyBundle(Buffer.alloc(0), Buffer.alloc(1000)), /actual 0 bytes SHA256.*expected 1000 bytes SHA256/);
  });
  test("equal subarray views respect their byte offsets", () => {
    const a = Uint8Array.from([0, 2, 3, 9]), b = Uint8Array.from([8, 2, 3, 7]);
    assert.equal(verifyBundle(a.subarray(1, 3), b.subarray(1, 3)).bytes, 2);
  });
  test("different subarray views cannot be hidden by a shared backing buffer", () => {
    const bytes = Uint8Array.from([1, 2, 3]);
    assert.throws(() => verifyBundle(bytes.subarray(0, 2), bytes.subarray(1, 3)), /mismatch/);
  });
  test("non-byte input fails with a plain type error", () => {
    assert.throws(() => verifyBundle(/** @type {Uint8Array} */ (/** @type {unknown} */ ([1])), Buffer.alloc(0)), TypeError);
  });
});
