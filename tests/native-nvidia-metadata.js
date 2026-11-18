"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs");
const os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {nvmlMetadataInput} = require("../scripts/build-native-application");
const repo = path.resolve(__dirname, "..");

test.describe("Native NVIDIA SDK-typed metadata and owned driver lifecycle", {skip: process.platform !== "linux"}, () => {
  const fixture = {root: ""};
  test.before(() => {
    fixture.root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-nvidia-"));
    for (const variant of ["enabled", "stub"]) {
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
        "-static-libstdc++", "-static-libgcc", "-I", repo, "-I", path.join(__dirname, "native/nvml"),
        ...(variant === "enabled" ? ["-DMOM_NATIVE_NVML_METADATA"] : []), path.join(repo, "runtime/nvidia-metadata.cpp"),
        path.join(__dirname, "native/nvidia-metadata.cpp"), "-o", path.join(fixture.root, variant)], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test.after(() => {if (fixture.root) {fs.rmSync(fixture.root, {recursive: true, force: true});}});
  for (const mode of ["mixed", "twice", "missing-driver", "missing-symbol", "init-error", "count-error", "zero", "huge-count", "handle-error",
    "null-handle", "second-error", "unset", "negative-major", "zero-major", "negative-minor", "bad-minor", "overflow", "shutdown-error", "throw", "stub"]) {
    test(`${mode}: conservative results, exact driver identity and balanced owned initialization`, () => {
      const result = spawnSync(path.join(fixture.root, mode === "stub" ? "stub" : "enabled"), [mode], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
      assert.match(result.stdout, /cleanup passed; no real GPU driver/);
    });
  }
});

test.describe("Native NVIDIA policy without an external utility", () => {
  /** @param {string} body */
  function run(body) {
    const prelude = 'const cp=require("node:child_process");cp.spawnSync=()=>{throw new Error("forbidden utility")};' +
      `const assert=require("node:assert/strict");const policy=require(${JSON.stringify(path.join(repo, "compiler-policy.js"))});`;
    const result = spawnSync(process.execPath, ["--max-old-space-size=128", "-e", prelude + body], {encoding: "utf8", timeout: 10000,
      env: {...process.env, PATH: "/unavailable", MOM_NVIDIA_COMPUTE_CAPABILITY: undefined}});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  }
  test("mixed physical capabilities retain the conservative minimum despite visibility/index masks", () => {
    run("global.__momNativeCore={nvidiaComputeCapabilities:()=>[120,70]};assert.equal(policy.nvidiaComputeCapability({CUDA_VISIBLE_DEVICES:'0',MOM_GPU_INDEX:'1'}),70);");
  });
  test("native capability queries are cached once per process", () => {
    run("let calls=0;global.__momNativeCore={nvidiaComputeCapabilities:()=>{calls++;return [120]}};assert.equal(policy.nvidiaComputeCapability({}),120);assert.equal(policy.nvidiaComputeCapability({}),120);assert.equal(calls,1);");
  });
  test("an older native bridge without metadata does not fall back to spawning a utility", () => {
    run("global.__momNativeCore={};assert.equal(policy.nvidiaComputeCapability({}),null);");
  });
  test("an empty native query remains unknown", () => {run("global.__momNativeCore={nvidiaComputeCapabilities:()=>[]};assert.equal(policy.nvidiaComputeCapability({}),null);");});
  test("non-array results cannot select an optimistic architecture", () => {run("global.__momNativeCore={nvidiaComputeCapabilities:()=>120};assert.equal(policy.nvidiaComputeCapability({}),null);");});
  test("a partial or malformed list cannot discard a problematic physical device", () => {
    run("global.__momNativeCore={nvidiaComputeCapabilities:()=>[120,null]};assert.equal(policy.nvidiaComputeCapability({}),null);");
  });
  test("explicit valid overrides do not query hardware", () => {
    run("global.__momNativeCore={nvidiaComputeCapabilities:()=>{throw Error('must not query')}};assert.equal(policy.nvidiaComputeCapability({MOM_NVIDIA_COMPUTE_CAPABILITY:'6.1'}),61);");
  });
  test("invalid overrides fail before native driver access", () => {
    run("global.__momNativeCore={nvidiaComputeCapabilities:()=>{throw Error('must not query')}};assert.throws(()=>policy.nvidiaComputeCapability({MOM_NVIDIA_COMPUTE_CAPABILITY:'bad'}),/Invalid MOM_NVIDIA/);");
  });
  test("unexpected native exceptions remain visible", () => {
    run("global.__momNativeCore={nvidiaComputeCapabilities:()=>{throw Error('native allocation')}};assert.throws(()=>policy.nvidiaComputeCapability({}),/native allocation/);");
  });
  test("Engineering Mode retains its existing utility query and parsing", () => {
    run("let calls=0;cp.spawnSync=()=>{calls++;return {status:0,stdout:'12.0\\n7.0\\n'}};delete require.cache[require.resolve(" + JSON.stringify(path.join(repo, "compiler-policy.js")) + ")];const original=require(" + JSON.stringify(path.join(repo, "compiler-policy.js")) + ");assert.equal(original.nvidiaComputeCapability({}),70);assert.equal(calls,1);");
  });
  test("the native Core bridge forwards only its host capability primitive", () => {
    const core = require("../runtime/core"), saved = globalThis.__momNvidiaComputeCapabilities, values = [120, 70];
    try {globalThis.__momNvidiaComputeCapabilities = () => values; assert.equal(core.nvidiaComputeCapabilities(), values);} finally {
      if (saved) {globalThis.__momNvidiaComputeCapabilities = saved;} else {Reflect.deleteProperty(globalThis, "__momNvidiaComputeCapabilities");}
    }
  });
});

test.describe("Native NVIDIA metadata header admission", () => {
  let root = "";
  test.before(() => {root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvml-header-"));});
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test("CPU-only proof builds do not select an installed GPU SDK implicitly", () => {assert.equal(nvmlMetadataInput(undefined), null);});
  for (const value of ["", "relative/nvml.h", 1, "/sdk/nvml.h\n", "/sdk/nvml.h\0"]) {
    test(`${JSON.stringify(value)} is rejected before header access`, () => {assert.throws(() => nvmlMetadataInput(value), /absolute NVML/);});
  }
  test("directories are not compiler-header inputs", () => {assert.throws(() => nvmlMetadataInput(root), /bounded regular/);});
  test("a modified/unreviewed header fails the content pin", () => {
    const file = path.join(root, "nvml.h"); fs.writeFileSync(file, "unreviewed\n");
    assert.throws(() => nvmlMetadataInput(file), /reviewed 12\.6\.77/);
  });
  test("oversized SDK inputs are rejected before reading a large buffer", () => {
    const file = path.join(root, "large.h"), fd = fs.openSync(file, "w");
    try {fs.ftruncateSync(fd, (2 << 20) + 1);} finally {fs.closeSync(fd);}
    assert.throws(() => nvmlMetadataInput(file), /bounded regular/);
  });
});
