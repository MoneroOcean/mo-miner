"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const environment = require("../runtime/environment");
const {extractCppBlock} = require("./common/cpp_source");

test.describe("Native process environment", {concurrency: false}, () => {
  /** @type {Record<string, unknown>} */
  const saved = {};
  const names = ["__momGetEnv", "__momSetEnv", "__momEnvKeys"];
  /** @type {Map<string, string>} */
  let values;
  /** @type {ReturnType<typeof environment>} */
  let env;
  test.beforeEach(() => {
    for (const name of names) {saved[name] = Reflect.get(globalThis, name);}
    values = new Map([["EXISTING", "initial"]]);
    globalThis.__momGetEnv = (name) => values.get(name);
    globalThis.__momSetEnv = (name, value) => {if (value === null) {values.delete(name);} else {values.set(name, value);}};
    globalThis.__momEnvKeys = () => [...values.keys()];
    env = environment();
  });
  test.afterEach(() => {
    for (const name of names) {
      if (saved[name] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[name]);}
    }
  });
  test("reads reflect current native state, not a stale startup snapshot", () => {
    assert.equal(Reflect.get(env, "EXISTING"), "initial"); values.set("EXISTING", "changed");
    assert.equal(Reflect.get(env, "EXISTING"), "changed");
  });
  test("string writes reach the OS bridge", () => {
    Reflect.set(env, "WORKER", "1"); assert.equal(values.get("WORKER"), "1");
  });
  test("numbers, booleans, null and undefined retain Node's string coercion", () => {
    for (const value of [4, false, null, undefined]) {
      Reflect.set(env, "VALUE", value); assert.equal(values.get("VALUE"), String(value));
    }
  });
  test("symbols cannot silently become environment values", () => {
    assert.throws(() => Reflect.set(env, "VALUE", Symbol("invalid")), TypeError);
    assert.throws(() => Reflect.set(env, Symbol("invalid"), "value"), TypeError);
    assert.equal(values.has("VALUE"), false);
  });
  test("enumeration and spread copy actual current worker environment", () => {
    Reflect.set(env, "GPU", "nvidia"); assert.deepEqual({...env}, {EXISTING: "initial", GPU: "nvidia"});
    values.delete("EXISTING"); assert.deepEqual(Object.keys(env), ["GPU"]);
  });
  test("enumeration preserves native order and handles an empty native list", () => {
    values = new Map([["ORDINARY", "ordinary"], ["UNICODE_é_💎", "unicode"], ["__proto__", "prototype"], ["toString", "method"]]);
    assert.deepEqual(Reflect.ownKeys(env), [...values.keys()]);
    values.clear(); assert.deepEqual(Reflect.ownKeys(env), []);
  });
  test("Object.assign applies each worker environment change to the native bridge", () => {
    Object.assign(env, {GPU: "amd", THREAD: 2}); assert.equal(values.get("GPU"), "amd");
    assert.equal(values.get("THREAD"), "2");
  });
  test("deletion removes the native variable without leaving a JS shadow", () => {
    assert.equal(Reflect.deleteProperty(env, "EXISTING"), true);
    assert.equal(values.has("EXISTING"), false); assert.equal(Reflect.get(env, "EXISTING"), undefined);
  });
  test("own-property descriptors reflect empty strings and disappear after unset", () => {
    Reflect.set(env, "EMPTY", "");
    assert.deepEqual(Object.getOwnPropertyDescriptor(env, "EMPTY"), {
      value: "", writable: true, enumerable: true, configurable: true,
    });
    Reflect.deleteProperty(env, "EMPTY"); assert.equal(Object.getOwnPropertyDescriptor(env, "EMPTY"), undefined);
  });
  test("valid defineProperty delegates to native state, not the proxy handler", () => {
    Object.defineProperty(env, "DEFINED", {value: 7, writable: true, enumerable: true, configurable: true});
    assert.equal(Reflect.get(env, "DEFINED"), "7"); assert.equal(Object.hasOwn(env, "DEFINED"), true);
  });
  test("accessors and restricted descriptors are rejected before mutation", () => {
    const target = env;
    for (const descriptor of [{value: 1}, {get() {return "wrong";}},
      {value: 1, writable: true, enumerable: true, configurable: false}]) {
      assert.throws(() => Object.defineProperty(target, "INVALID", descriptor),
        {code: "ERR_INVALID_OBJECT_DEFINE_PROPERTY"});
    }
    assert.equal(values.has("INVALID"), false);
  });
  test("prototype-named variables cannot modify the environment object's prototype", () => {
    const prototype = Object.getPrototypeOf(env);
    Reflect.set(env, "__proto__", "worker-data"); assert.equal(Reflect.get(env, "__proto__"), "worker-data");
    assert.equal(Object.getPrototypeOf(env), prototype); assert.equal(Object.hasOwn(env, "__proto__"), true);
  });
  test("inherited object methods remain available unless overridden by a variable", () => {
    assert.equal(typeof Reflect.get(env, "toString"), "function");
    assert.equal("EXISTING" in env, true); assert.equal("MISSING" in env, false);
    Reflect.set(env, "toString", "overridden"); assert.equal(Reflect.get(env, "toString"), "overridden");
  });
  test("OS failures are surfaced without claiming a successful environment mutation", () => {
    globalThis.__momSetEnv = () => {throw new Error("OS failed");};
    assert.throws(() => Reflect.set(env, "GPU", "intel"), /OS failed/); assert.equal(values.has("GPU"), false);
  });
  test("the actual binding filters Windows hidden names without changing other platforms", {
    timeout: 30000,
    // Like other SDK-free C++ source fixtures, compile both OS branches with a host GNU driver.
    // Windows itself is covered by the direct-executable parity run, not a MinGW prerequisite.
    skip: process.platform === "win32",
  }, (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-environment-enumeration-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const runtime = fs.readFileSync(path.join(__dirname, "../runtime/event-loop.cpp"), "utf8");
    const fixture = fs.readFileSync(path.join(__dirname, "native/environment_enumeration.cpp"), "utf8");
    const declaration = '  binding(*owner, "__momEnvKeys", 0, [](SHConsoleContext &owner, const Value *) -> Value {';
    const body = extractCppBlock(runtime, declaration).replace(declaration, "Array enumerate(Owner &owner) {");
    assert.equal((fixture.match(/\/\* MOM_ENV_ENUMERATION_BODY \*\//g) || []).length, 1);
    const source = fixture.replace("/* MOM_ENV_ENUMERATION_BODY */", body);
    for (const windows of [false, true]) {
      const executable = path.join(root, windows ? "windows" : "portable");
      const compile = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        ...(windows ? ["-D_WIN32"] : []), "-x", "c++", "-", "-o", executable],
      {input: source, encoding: "utf8", timeout: 8000, maxBuffer: 4 << 20});
      assert.equal(compile.status, 0, compile.error?.message || compile.stderr);
      const run = spawnSync(executable, [], {encoding: "utf8", timeout: 4000, maxBuffer: 1 << 20});
      assert.equal(run.status, 0, run.error?.message || run.stderr);
      assert.equal(run.stdout, "native environment enumeration passed\n");
    }
  });
  test("native environment updates synchronize Windows CRT selection without changing libuv semantics", {
    timeout: 30000, skip: process.platform === "win32",
  }, (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-environment-update-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(path.join(root, "uv.h"), "#pragma once\n#define UV_EINVAL (-22)\n#define UV_EIO (-5)\n");
    for (const windows of [false, true]) {
      const executable = path.join(root, windows ? "windows" : "portable");
      const compile = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra",
        ...(windows ? ["-D_WIN32"] : []), "-I", root, "-I", path.resolve(__dirname, ".."),
        path.join(__dirname, "native/environment_update.cpp"), "-o", executable],
      {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      assert.equal(compile.status, 0, compile.error?.message || compile.stderr);
      for (let mode = 0; mode < 10; mode++) {
        const run = spawnSync(executable, [String(mode)], {encoding: "utf8", timeout: 3000, maxBuffer: 1 << 20});
        assert.equal(run.status, 0, `${windows ? "Windows" : "portable"} case ${mode}: ${run.error?.message || run.stderr}`);
      }
      if (windows) {
        const baseline = path.join(root, "old-os-only");
        const before = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-D_WIN32", "-DMOM_ENV_OS_ONLY_BASELINE",
          "-I", root, "-I", path.resolve(__dirname, ".."), path.join(__dirname, "native/environment_update.cpp"), "-o", baseline],
        {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
        assert.equal(before.status, 0, before.error?.message || before.stderr);
        assert.equal(spawnSync(baseline, ["0"], {timeout: 3000}).status, 3, "Old libuv-only update must expose the stale CRT");
      }
    }
    const binding = fs.readFileSync(path.join(__dirname, "../runtime/event-loop.cpp"), "utf8");
    assert.match(binding, /const int status = setEnvironment\(name\.c_str\(\), args\[1\]\.isNull\(\) \? nullptr : value\.c_str\(\)\)/);
  });
});
