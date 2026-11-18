"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"),
  test = require("node:test"), {createHash} = require("node:crypto");
const {prepareNativePackingTool} = require("../scripts/prepare-native-packing-tool");
const toolCatalog = require("../scripts/native-packing-tool.json");

// Inert fixture bytes stand in for the pinned archive and executable.
const toolArchive = "fixture upx archive", toolBytes = "fixture upx";
const digest = (/** @type {string} */ value) => createHash("sha256").update(value).digest("hex");

test.describe("pinned packing tool acquisition", {skip: (process.platform !== "linux" || process.arch !== "x64") && "Linux x64 packing host"}, () => {
  test("a failed download leaves nothing behind and the next run acquires without manual cleanup", t => {
    const saved = JSON.stringify({archive: toolCatalog.archive, executable: toolCatalog.executable});
    Object.assign(toolCatalog.archive, {bytes: Buffer.byteLength(toolArchive), sha256: digest(toolArchive)});
    Object.assign(toolCatalog.executable, {bytes: Buffer.byteLength(toolBytes), sha256: digest(toolBytes)});
    const temp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-packing-tool-"));
    t.after(() => {
      const original = JSON.parse(saved);
      Object.assign(toolCatalog.archive, original.archive); Object.assign(toolCatalog.executable, original.executable);
      fs.rmSync(temp, {recursive: true, force: true});
    });
    const output = path.join(temp, "packing-tool"), curls = /** @type {string[][]} */ ([]);
    const spawn = /** @type {any} */ ((/** @type {string} */ command, /** @type {string[]} */ args, /** @type {any} */ options) => {
      if (command === "tar") {fs.writeSync(options.stdio[1], toolBytes); return {status: 0};}
      curls.push(args);
      // The first transfer dies after writing a truncated archive.
      fs.writeFileSync(args[args.indexOf("--output") + 1] || "", curls.length === 1 ? "fixture" : toolArchive);
      return {status: curls.length === 1 ? 56 : 0};
    });
    assert.throws(() => prepareNativePackingTool(output, {}, spawn), /acquisition failed: curl/);
    assert.deepEqual(fs.readdirSync(temp), []);
    const tool = prepareNativePackingTool(output, {}, spawn);
    assert.equal(tool.upx, path.join(output, "upx"));
    assert.deepEqual(fs.readdirSync(output).sort(), ["upx", "upx.tar.xz"]);
    assert.deepEqual(fs.readdirSync(temp), ["packing-tool"]);
    assert.ok(curls.every(args => args.includes("--retry")), "curl retries transient transfer failures itself");
    assert.equal(prepareNativePackingTool(output, {}, spawn).upx, tool.upx);
    assert.equal(curls.length, 2, "an admitted tool directory is reused");
  });
});
