"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const {runInNewContext} = require("node:vm");
const policy = require("../compiler-policy");
const helper = require("../helper");

const repo = path.resolve(__dirname, "..");
const packaged = path.join(repo, "libs");
const legacy = path.join(repo, "build", "win", "Release");
const source = path.join(repo, "build", "win", "compilers");

/** @param {import("node:test").TestContext} t @param {string[]} files */
function available(t, files) {
  const paths = new Set(files);
  t.mock.method(fs, "existsSync", (/** @type {import("node:fs").PathLike} */ file) =>
    paths.has(String(file)));
}

/** @param {string} gpu @param {NodeJS.ProcessEnv} [extra] @returns {NodeJS.ProcessEnv} */
function environment(gpu, extra = {}) {
  return {MOM_GPU_BACKEND: gpu, MOM_NVIDIA_COMPUTE_CAPABILITY: "86", ...extra};
}

/** @param {string} root @param {string} compiler */
function addon(root, compiler) {
  return path.join(root, compiler, "mom.node");
}

test.describe("Windows source worker discovery", {concurrency: false}, () => {
  test("plain Windows source loading prepares the oneAPI runtime without binding a vendor", (t) => {
    available(t, [addon(source, "oneapi"), path.join(legacy, "mom.node")]);
    const env = {Path: "original"};
    assert.equal(helper.prepare_native_core_path(env, "win32"), addon(source, "oneapi"));
    assert.equal(env.Path, [path.join(source, "oneapi"), "original"].join(path.delimiter));
    assert.equal(Object.hasOwn(env, "MOM_GPU_BACKEND"), false);
    assert.equal(Object.hasOwn(env, "MOM_NATIVE_PATH"), false);
  });

  test("default loading preserves packaged precedence over Windows source outputs", (t) => {
    const packagedAddon = path.join(path.dirname(process.execPath), "libs", "mom.node");
    available(t, [packagedAddon, addon(source, "oneapi")]);
    const env = {Path: "original"};
    assert.equal(helper.prepare_native_core_path(env, "win32"), packagedAddon);
    assert.equal(env.Path, [path.dirname(packagedAddon), "original"].join(path.delimiter));
  });

  test("explicit addon loading prepares only its own runtime even for missing workers", (t) => {
    t.mock.method(fs, "existsSync", () => {throw new Error("Unexpected fallback lookup");});
    for (const compiler of ["dpcpp", "acpp-hip"]) {
      const native = addon(source, compiler);
      const env = {MOM_NATIVE_PATH: native, Path: "original", PATH: "duplicate"};
      assert.equal(helper.prepare_native_core_path(env, "win32"), native);
      assert.equal(env.Path, [path.dirname(native), "original"].join(path.delimiter));
      assert.equal(Object.hasOwn(env, "PATH"), false);
      assert.ok(!env.Path.includes(path.join(source, "oneapi")));
    }
  });

  test("plain source loading retains legacy Windows and Linux fallback behavior", (t) => {
    const linuxAddon = path.join(repo, "build", "lin", "Release", "mom.node");
    const windowsAddon = path.join(legacy, "mom.node");
    available(t, [windowsAddon, linuxAddon, addon(source, "oneapi")]);
    const linuxEnv = {PATH: "original"};
    assert.equal(helper.prepare_native_core_path(linuxEnv, "linux"), linuxAddon);
    assert.equal(linuxEnv.PATH, "original");
    available(t, [windowsAddon]);
    const windowsEnv = {Path: "original"};
    assert.equal(helper.prepare_native_core_path(windowsEnv, "win32"), windowsAddon);
    assert.equal(windowsEnv.Path, [legacy, "original"].join(path.delimiter));
  });

  test("packaged native addon contracts do not import the omitted source helper", () => {
    const contracts = fs.readFileSync(path.join(repo, "tests", "native_addon.js"), "utf8");
    const start = contracts.indexOf("  const root =");
    const end = contracts.indexOf("  const core =", start);
    assert.ok(start > 0 && end > start);
    const native = addon(packaged, "acpp-hip");
    const selected = runInNewContext(contracts.slice(start, end) + "\naddonPath;", {
      __dirname, path, fs: {existsSync: () => false},
      process: {platform: "win32", env: {MOM_NATIVE_PATH: native}},
      require: () => {throw new Error("Package imported an omitted source helper");},
    });
    assert.equal(selected, native);
  });

  test("oneAPI source export carries its matched dynamically loaded runtime and DLL closure", () => {
    const builder = fs.readFileSync(path.join(repo, ".github", "workflows", "scripts",
      "build-windows-multicompiler.ps1"), "utf8");
    const start = builder.indexOf("      Save-Compiler oneapi\n");
    const end = builder.indexOf("\n    # Targeted Intel builds", start);
    assert.ok(start > 0 && end > start);
    const exportSource = builder.slice(start, end);
    assert.match(exportSource, /Copy-MominerOptionalRuntimeFiles -PackageDir \$oneApiDest -SourceRoots @\(Get-MominerOneApiBinDirs\)/);
    assert.match(exportSource, /Copy-MominerDllClosure -PackageDir \$oneApiDest -EntryPaths \$oneApiEntries/);
    assert.ok(exportSource.indexOf("Remove-Item Env:MOM_DPCPP_DIR, Env:MOM_ACPP_DIR") <
      exportSource.indexOf("Copy-MominerDllClosure"));
    assert.match(exportSource, /finally \{[\s\S]*\$env:MOM_DPCPP_DIR = \$oneApiSavedDpcpp[\s\S]*\$env:MOM_ACPP_DIR = \$oneApiSavedAcpp/);
    const runtimeHelper = fs.readFileSync(path.join(repo, ".github", "workflows", "scripts",
      "windows-dll-deps.ps1"), "utf8");
    const patterns = runtimeHelper.slice(runtimeHelper.indexOf("function Copy-MominerOptionalRuntimeFiles"),
      runtimeHelper.indexOf("function Copy-MominerHipDynamicRuntimeFiles"));
    assert.match(patterns, /'sycl\*\.dll'/);
    assert.match(patterns, /'ur_\*\.dll'/);
    assert.match(patterns, /Test-MominerDebugRuntimeDllName/);
    assert.doesNotMatch(exportSource, /Copy-Item.*-Recurse/);
  });

  for (const [gpu, algo, compiler] of [
    ["intel", "etchash", "oneapi"],
    ["nvidia", "etchash", "dpcpp"],
    ["nvidia", "autolykos2", "acpp-cuda"],
    ["amd", "etchash", "acpp-hip"],
    ["intel_igpu", "etchash", "dpcpp-opencl"],
  ]) {
    test(`discovers built ${compiler} worker for ${gpu}/${algo}`, (t) => {
      assert.ok(gpu && algo && compiler);
      const native = addon(source, compiler);
      available(t, [native]);
      const result = policy.workerEnv(algo, environment(gpu), "win32");
      assert.equal(result["MOM_NATIVE_PATH"], native);
      assert.equal(result["MOM_RUNTIME_DIR"], path.dirname(native));
      assert.equal(result["MOM_SYCL_COMPILER"], compiler);
      const runtimePaths = result["PATH"]?.split(path.delimiter);
      assert.equal(runtimePaths?.[0], path.dirname(native));
      if (compiler === "dpcpp-opencl") {
        assert.equal(runtimePaths?.[2], path.join(source, "oneapi"));
      }
    });
  }

  test("source control discovery uses the policy default without an algorithm override", (t) => {
    available(t, [addon(source, "dpcpp"), addon(source, "acpp-cuda")]);
    const result = policy.workerEnv("__control__", environment("nvidia"), "win32");
    assert.equal(result["MOM_NATIVE_PATH"], addon(source, "dpcpp"));
  });

  test("packaged workers and shared runtime never cross into source outputs", (t) => {
    available(t, [addon(packaged, "dpcpp-opencl"), addon(source, "dpcpp-opencl"),
      addon(source, "oneapi")]);
    const result = policy.workerEnv("etchash", environment("intel_igpu", {Path: "existing"}), "win32");
    assert.equal(result["MOM_NATIVE_PATH"], addon(packaged, "dpcpp-opencl"));
    assert.deepEqual(result["Path"]?.split(path.delimiter), [
      path.join(packaged, "dpcpp-opencl"), path.join(packaged, "dpcpp-opencl", "hipSYCL"),
      path.join(packaged, "oneapi"), "existing",
    ]);
  });

  test("legacy Release worker retains precedence over multicompiler outputs", (t) => {
    available(t, [addon(legacy, "oneapi"), addon(source, "oneapi")]);
    assert.equal(policy.workerEnv("etchash", environment("intel"), "win32")["MOM_NATIVE_PATH"],
      addon(legacy, "oneapi"));
  });

  test("explicit native directory retains precedence and matching runtime", (t) => {
    const custom = path.join(repo, "custom-workers");
    available(t, [addon(custom, "oneapi"), addon(packaged, "oneapi"), addon(source, "oneapi")]);
    const result = policy.workerEnv("etchash", environment("intel", {MOM_NATIVE_DIR: custom}), "win32");
    assert.equal(result["MOM_NATIVE_PATH"], addon(custom, "oneapi"));
    assert.equal(result["MOM_RUNTIME_DIR"], path.join(custom, "oneapi"));
  });

  test("a missing native directory keeps existing policy fallback semantics", (t) => {
    available(t, [addon(source, "oneapi")]);
    assert.equal(policy.workerEnv("etchash", environment("intel", {
      MOM_NATIVE_DIR: path.join(repo, "missing-workers"), MOM_COMPILER_POLICY_STRICT: "1",
    }), "win32")["MOM_NATIVE_PATH"], addon(source, "oneapi"));
  });

  test("an explicit native path remains authoritative even if missing", (t) => {
    available(t, [addon(source, "oneapi")]);
    const env = environment("intel", {MOM_NATIVE_PATH: path.join(repo, "explicit-missing.node"),
      MOM_COMPILER_POLICY_STRICT: "1"});
    const result = {...env, ...policy.workerEnv("etchash", env, "win32")};
    assert.equal(result["MOM_NATIVE_PATH"], env["MOM_NATIVE_PATH"]);
    assert.equal(result["MOM_RUNTIME_DIR"], undefined);
  });

  test("launcher default can switch to an algorithm-specific source worker", (t) => {
    const control = addon(source, "dpcpp");
    available(t, [control, addon(source, "acpp-cuda")]);
    const result = policy.workerEnv("autolykos2", environment("nvidia", {
      MOM_NATIVE_PATH: control, MOM_NATIVE_PATH_LAUNCHER_DEFAULT: control,
    }), "win32");
    assert.equal(result["MOM_NATIVE_PATH"], addon(source, "acpp-cuda"));
  });

  test("missing selected compiler does not borrow a foreign worker or runtime", (t) => {
    available(t, [addon(source, "oneapi"), addon(source, "dpcpp")]);
    const env = environment("amd");
    assert.equal(policy.workerEnv("etchash", env, "win32")["MOM_NATIVE_PATH"], undefined);
    assert.throws(() => policy.workerEnv("etchash", {...env, MOM_COMPILER_POLICY_STRICT: "1"},
      "win32"), /Missing acpp-hip\/mom\.node for amd\/etchash/);
    assert.throws(() => policy.workerEnv("etchash", environment("intel_igpu"), "win32"),
      /Missing dpcpp-opencl\/mom\.node/);
    assert.throws(() => policy.workerEnv("etchash", environment("nvidia", {
      MOM_NVIDIA_COMPUTE_CAPABILITY: "75",
    }), "win32"), /SM75 cannot use the SM80-only default worker/);
  });

  test("Linux does not search Windows or a new Linux compilers directory", (t) => {
    available(t, [addon(source, "oneapi"), addon(path.join(repo, "build", "lin", "compilers"), "oneapi")]);
    assert.throws(() => policy.workerEnv("etchash", environment("intel", {
      MOM_COMPILER_POLICY_STRICT: "1",
    }), "linux"), /Missing oneapi\/mom\.node/);
  });
});
