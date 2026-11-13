"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {stageWindowsDpcppComponent} = require("../../scripts/native-windows-dpcpp-component");
const required = ["ur_loader", "ur_common", "ur_adapter_level_zero", "ur_adapter_opencl", "level_zero_utils", "umf", "umf_ba", "umf_coarse", "umf_utils", "hwloc", "OpenCL", "cuda", "level-zero-v2", "sycl-jit"];

/** Structural COFF payload only; these fixtures cannot qualify a compiler or miner. */
function coffObject() {
  const b = Buffer.alloc(61); b.writeUInt16LE(0x8664); b.writeUInt16LE(1, 2); b.write(".text", 20); b.writeUInt32LE(1, 36);
  b.writeUInt32LE(60, 40); b.writeUInt32LE(0x60000020, 56); b[60] = 0xc3; return b;
}

/** One tiny complete component for transport and full-application admission tests.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-component-")), input = path.join(root, "input"), output = path.join(root, "installed");
  fs.mkdirSync(input); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const library = (/** @type {string} */ n) => {const p = path.join(input, n + ".lib"); fs.writeFileSync(p, "!<arch>\n"); return p;};
  const objects = ["drivers.obj", "driver-aliases.obj"].map(n => {const p = path.join(input, n); fs.writeFileSync(p, coffObject()); return p;});
  const links = {libraries: [...required, ...Array.from({length: 50}, (_, n) => "LLVM" + n)].map(library), wholeArchives: ["mom-sycl-static", "ze_loader"].map(library), objects};
  const produce = () => stageWindowsDpcppComponent(links, {path: "/original/receipt.json", sha256: "1".repeat(64)}, output);
  const edit = (/** @type {(r: any) => void} */ change) => {const p = path.join(output, "component.json"), r = JSON.parse(fs.readFileSync(p, "utf8")); change(r); fs.writeFileSync(p, JSON.stringify(r));};
  return {root, input, output, links, produce, edit};
}

/** Tiny admitted-shape vendor SDK for reader/application tests; no vendor bytes or claims of qualification.
 * @param {string} parent */
function vendorNvrtcFixture(parent) {
  const {fileHash} = require("../../scripts/build-source-inputs"), recipeFiles = require("../../scripts/prepare-native-nvrtc-sdk").nvrtcSdkRecipe().graph,
    root = path.join(parent, "vendor-nvrtc"), originals = path.join(parent, "nvrtc-originals");
  /** @type {Record<string, {kind: "file", bytes: number, sha256: string, source: {archive: "nvrtc" | "nvptxcompiler", member: string}}>} */
  const outputs = {};
  /** @type {Record<string, string>} */ const files = {};
  fs.mkdirSync(path.join(root, "lib/x64"), {recursive: true}); fs.mkdirSync(path.join(root, "include"));
  fs.mkdirSync(path.join(root, "licenses/nvrtc"), {recursive: true}); fs.mkdirSync(path.join(root, "licenses/nvptxcompiler")); fs.mkdirSync(originals);
  const members = ["include/nvrtc.h", "include/nvPTXCompiler.h", "lib/x64/nvrtc_static.lib", "lib/x64/nvrtc-builtins_static.lib",
    "lib/x64/nvptxcompiler_static.lib", "licenses/nvrtc/LICENSE", "licenses/nvptxcompiler/LICENSE"];
  for (const member of members) {
    const file = path.join(root, ...member.split("/")), data = member.endsWith(".lib") ? "!<arch>\n" : "fixture " + member + "\n";
    fs.writeFileSync(file, data);
    const archive = member.includes("nvptxcompiler") ? "nvptxcompiler" : "nvrtc";
    outputs[member] = {kind: "file", bytes: fs.statSync(file).size, sha256: fileHash(file), source: {archive, member}};
    files[file] = outputs[member].sha256;
  }
  /** @type {Record<string, {path: string, url: string, sha256: string, bytes: number}>} */ const archives = {};
  for (const name of ["nvrtc", "nvptxcompiler"]) {
    const file = path.join(originals, name + ".zip"); fs.writeFileSync(file, "fixture archive " + name);
    archives[name] = {path: file, url: "https://vendor.invalid/" + name + ".zip", sha256: fileHash(file), bytes: fs.statSync(file).size};
    files[file] = archives[name]?.sha256 || "";
  }
  const sourceReceipt = path.join(originals, "fetch.json"), tool = path.join(originals, "unzip");
  fs.writeFileSync(sourceReceipt, JSON.stringify({archives})); fs.writeFileSync(tool, "fixture unzip tool");
  files[sourceReceipt] = fileHash(sourceReceipt); files[tool] = fileHash(tool);
  const recipeRoot = path.resolve(__dirname, "../.."), receipt = {version: 1, target: "nvrtc-sdk-windows", platform: "windows", vendorVersion: "13.1.115",
    status: "PASS", networked: false, buildOnly: true, compiled: false, sourceRebuilt: false, vendorPrebuilt: true,
    publicationQualified: false, redistributionQualified: false, producerSha256: "a".repeat(64), recipe: {sha256: "b".repeat(64), files: recipeFiles},
    tool: {path: tool, sha256: fileHash(tool), version: "fixture unzip"},
    source: {directory: originals, receipt: {version: 1, target: "nvrtc-windows", status: "PASS", networked: true, buildOnly: true, compiled: false,
      publicationQualified: false, fetcherSha256: "c".repeat(64), recipe: {sha256: "d".repeat(64), files: {}}, archives},
    receiptSha256: fileHash(sourceReceipt)}, output: root, files: outputs};
  const receiptFile = path.join(root, "sdk.json"); fs.writeFileSync(receiptFile, JSON.stringify(receipt));
  const receiptSha256 = fileHash(receiptFile); files[receiptFile] = receiptSha256;
  for (const [relative, hash] of Object.entries(recipeFiles)) {files[path.resolve(recipeRoot, ...relative.split("/"))] = hash;}
  return {root, libraries: members.filter(member => member.startsWith("lib/x64/")).map(member => path.join(root, ...member.split("/"))),
    receipt, receiptFile, receiptSha256, files};
}

module.exports = {required, coffObject, fixture, vendorNvrtcFixture};
