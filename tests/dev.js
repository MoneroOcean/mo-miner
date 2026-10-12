"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {runDev, windowsDevCommand, nativeDevEnvironment, windowsRuntimeEnvironment} = require("../scripts/dev");

test.describe("Cross-platform development dispatcher", () => {
  test("Linux delegates every argument and environment unchanged to the container entrypoint", () => {
    const env = {MOM_GPU_BACKEND: "caller-policy", MOM_REUSE_BUILT_WORKER: "caller-policy"};
    const args = ["npm", "test", "--", "cpu", '--config={"name":"quoted"}'];
    let calls = 0;
    assert.equal(runDev("linux", "x64", env, args, (command, forwarded, options) => {
      calls++;
      assert.equal(command, "bash");
      assert.equal(forwarded[0], path.resolve(__dirname, "../scripts/dev-container.sh"));
      assert.deepEqual(forwarded.slice(1), args);
      assert.equal(options.env, env);
      assert.equal(options.stdio, "inherit");
      assert.equal(options.shell, undefined);
      return {status: 17};
    }), 17);
    assert.equal(calls, 1);
  });

  test("Windows builds selected workers incrementally then launches node without shell interpretation", () => {
    const env = {MOM_GPU_BACKEND: "nvidia", MOM_BUILD_JOBS: "4", Path: "C:\\Tools\\node;C:\\Windows\\System32"};
    const args = ["node", "mom.js", '--config={"path":"C:\\a b","value":"& % !"}'];
    let calls = 0;
    assert.equal(runDev("win32", "x64", env, args, (command, forwarded, options) => {
      if (calls++ === 0) {
        assert.equal(command, "powershell.exe");
        assert.ok(forwarded.includes("-File"));
        assert.ok(forwarded.some(arg => arg.endsWith("build-windows-multicompiler.ps1")));
        assert.deepEqual(forwarded.slice(-2), ["-Backend", "nvidia"]);
        assert.equal(options.env?.["Path"], env.Path);
        assert.equal(options.env?.["MOM_NATIVE_PATH"], undefined);
        assert.equal(forwarded.some(arg => /package|docker|run\.sh|build\/dev/.test(arg)), false);
      } else {
        assert.equal(command, process.execPath);
        assert.deepEqual(forwarded, args.slice(1));
        assert.ok(options.env?.["MOM_NATIVE_DIR"]?.endsWith(path.join("build", "win", "compilers")));
        assert.equal(options.env?.["MOM_NATIVE_PATH"], undefined);
        assert.equal(options.env?.["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], undefined);
        assert.equal(options.env?.["MOM_GPU_TEST_VENDORS"], "nvidia");
        assert.ok(options.env?.["Path"]?.endsWith(env.Path));
      }
      assert.equal(options.env?.["MOM_BUILD_JOBS"], "4");
      assert.equal(options.shell, undefined);
      return {status: 0};
    }), 0);
    assert.equal(calls, 2);
    assert.equal(env.Path, "C:\\Tools\\node;C:\\Windows\\System32");
  });

  test("npm uses its exact executable script and preserves Windows paths, spaces and JSON", () => {
    const env = {npm_execpath: "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js"};
    const args = ["npm", "test", "--", '{"name":"a b","path":"C:\\data","meta":"&%"}'];
    assert.deepEqual(windowsDevCommand(args, env), [process.execPath, env.npm_execpath, ...args.slice(1)]);
    assert.deepEqual(windowsDevCommand(["NODE.EXE", "mom.js"], {}), [process.execPath, "mom.js"]);
    assert.throws(() => windowsDevCommand(args, {}), /npm_execpath/);
    assert.throws(() => windowsDevCommand(args, {npm_execpath: "npm.cmd"}), /npm-cli\.js/);
    assert.deepEqual(windowsDevCommand(["C:\\Program Files\\tool.exe", "&", '"JSON"'], {}),
      ["C:\\Program Files\\tool.exe", "&", '"JSON"']);
    assert.deepEqual(windowsDevCommand(["C:\\Program Files\\tool.cmd", "a&b"], {}),
      ["cmd.exe", "/d", "/v:off", "/s", "/c", '""C:\\Program Files\\tool.cmd" "a&b""']);
    assert.throws(() => windowsDevCommand(["tool.bat", '"unsafe"'], {}), /safely preserve/);
  });

  test("no arguments builds then starts MoM on both platforms", () => {
    let calls = 0;
    assert.equal(runDev("win32", "x64", {}, [], (command, args) => {
      if (calls++ === 1) {
        assert.equal(command, process.execPath);
        assert.deepEqual(args, ["mom.js"]);
      }
      return {status: 0};
    }), 0);
    assert.equal(calls, 2);
    runDev("linux", "x64", {}, [], (command, args) => {
      assert.equal(command, "bash");
      assert.deepEqual(args.slice(1), ["node", "mom.js"]);
      return {status: 0};
    });
  });

  test("batch commands use verbatim cmd.exe arguments with the safe wrapper", () => {
    let calls = 0;
    runDev("win32", "x64", {}, ["--exec", "C:\\Program Files\\tool.cmd", "a&b"],
      (command, args, options) => {
        if (calls++ === 0) {return {status: 0};}
        assert.equal(command, "cmd.exe");
        assert.deepEqual(args, ["/d", "/v:off", "/s", "/c", '""C:\\Program Files\\tool.cmd" "a&b""']);
        assert.equal(options.windowsVerbatimArguments, true);
        assert.equal(options.shell, undefined);
        return {status: 0};
      });
  });

  test("build failures prevent command launch and command failures retain their exit status", () => {
    for (const status of [37, null]) {
      let calls = 0;
      assert.equal(runDev("win32", "x64", {}, ["node", "mom.js"], () => {
        calls++;
        return {status};
      }), status ?? 1);
      assert.equal(calls, 1);
    }
    let calls = 0;
    assert.equal(runDev("win32", "x64", {}, ["node", "mom.js"], () => ({status: calls++ ? 42 : 0})), 42);
    assert.throws(() => runDev("win32", "x64", {}, [], () => ({status: null, error: new Error("ENOENT")})),
      /Windows builder.*ENOENT.*install-dev\.bat/);
    assert.throws(() => runDev("win32", "x64", {}, ["npm", "test"], () => {
      assert.fail("invalid npm command must not reach the builder");
    }), /npm_execpath/);
  });

  test("Windows refuses unsafe unchecked reuse before starting any process", () => {
    assert.throws(() => runDev("win32", "x64", {MOM_REUSE_BUILT_WORKER: "1"}, ["node", "mom.js"],
      () => {assert.fail("unsupported reuse must not launch a process");}), /unsupported in native development.*validates incremental caches/);
  });

  test("backend validation and test vendor narrowing match the Linux container defaults", () => {
    assert.equal(nativeDevEnvironment({})["MOM_GPU_TEST_VENDORS"], undefined);
    assert.equal(nativeDevEnvironment({MOM_GPU_BACKEND: "INTEL_IGPU"})["MOM_GPU_TEST_VENDORS"], "intel");
    assert.equal(nativeDevEnvironment({MOM_GPU_BACKEND: "all"})["MOM_GPU_TEST_VENDORS"], "intel,nvidia,amd");
    assert.equal(nativeDevEnvironment({MOM_GPU_BACKEND: "opencl"})["MOM_GPU_TEST_VENDORS"], undefined);
    assert.equal(nativeDevEnvironment({MOM_GPU_BACKEND: "amd", MOM_GPU_TEST_VENDORS: "intel"})["MOM_GPU_TEST_VENDORS"], "intel");
    assert.throws(() => nativeDevEnvironment({MOM_GPU_BACKEND: "invalid"}), /Unknown MOM_GPU_BACKEND/);
    assert.throws(() => nativeDevEnvironment({MOM_REUSE_BUILT_WORKER: "yes"}), /must be 0 or 1/);
    assert.throws(() => runDev("darwin", "x64", {}, []), /supports Linux/);
    assert.throws(() => runDev("win32", "arm64", {}, []), /Windows x86-64/);
  });

  test("runtime environment leaves addon, SDK and PATH selection to the existing compiler policy", () => {
    const root = "C:\\Project Files\\mom\\build\\win\\compilers";
    for (const backend of ["intel", "all", "nvidia", "amd", "opencl", "intel_igpu"]) {
      const env = windowsRuntimeEnvironment({MOM_NATIVE_DIR: root, MOM_GPU_BACKEND: backend, Path: "caller-tools"});
      assert.equal(env["MOM_NATIVE_DIR"], root);
      assert.equal(env["MOM_NATIVE_PATH"], undefined);
      assert.equal(env["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], undefined);
      assert.equal(env["Path"], "caller-tools");
      assert.equal(env["PATH"], undefined);
    }
    const custom = {MOM_NATIVE_DIR: root, MOM_NATIVE_PATH: "C:\\Custom Runtime\\mom.node",
      MOM_NATIVE_PATH_LAUNCHER_DEFAULT: "caller-default", PATH: "caller-tools", Path: "caller-tools"};
    const env = windowsRuntimeEnvironment(custom);
    assert.equal(env["MOM_NATIVE_PATH"], custom.MOM_NATIVE_PATH);
    assert.equal(env["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], "caller-default");
    assert.equal(env["PATH"], "caller-tools");
    assert.equal(env["Path"], "caller-tools");
    assert.equal(custom.Path, "caller-tools");
    assert.equal(windowsRuntimeEnvironment({MOM_NATIVE_PATH: custom.MOM_NATIVE_PATH})["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"], undefined);
  });

  test("npm build normalization is idempotent without removing runtime command PATH", () => {
    const npmCli = "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js";
    const packageJson = "C:\\work\\mom\\package.json";
    const scaffold = ["C:\\work\\mom\\node_modules\\.bin", "C:\\work\\node_modules\\.bin",
      "C:\\node_modules\\.bin", "C:\\Program Files\\nodejs\\node_modules\\npm\\node_modules\\@npmcli\\run-script\\lib\\node-gyp-bin"];
    const env = {npm_execpath: npmCli, npm_package_json: packageJson, npm_lifecycle_event: "dev",
      Path: [...scaffold, "caller-tools"].join(";")};
    for (let iteration = 0; iteration < 2; iteration++) {
      let calls = 0;
      runDev("win32", "x64", env, ["npm", "test"], (_command, _args, options) => {
        assert.equal(options.env?.["Path"]?.endsWith(calls++ ? env.Path : "caller-tools"), true);
        if (calls === 1) {
          assert.equal(options.env?.["Path"], "caller-tools");
          assert.equal(options.env?.["npm_package_json"], undefined);
        }
        return {status: 0};
      });
    }
  });
});

test.describe("Native host development dispatch", () => {
  test("Linux host uses the shared source entrypoint and preserves command arguments", () => {
    const args = ["node", "mom.js", "--host", "--", '{"path":"a b","meta":"& % ! $ `"}'];
    /** @type {NodeJS.ProcessEnv} */
    const env = {MOM_DEV_HOST: "1", MOM_GPU_BACKEND: "nvidia", MOM_BUILD_JOBS: "4"};
    let calls = 0;
    assert.equal(runDev("linux", "x64", env, args, (command, forwarded, options) => {
      calls++;
      assert.equal(command, "bash");
      assert.equal(forwarded[0], path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"));
      assert.deepEqual(forwarded.slice(1), args);
      assert.equal(options.cwd, path.resolve(__dirname, ".."));
      assert.equal(options.env?.["MOM_DEV_HOST"], "1");
      assert.equal(options.env?.["MOM_GPU_BACKEND"], "nvidia");
      assert.equal(options.env?.["MOM_GPU_TEST_VENDORS"], "nvidia");
      assert.equal(options.env?.["MOM_BUILD_JOBS"], "4");
      assert.equal(options.shell, undefined);
      assert.equal(options.stdio, "inherit");
      return {status: 0};
    }), 0);
    assert.equal(calls, 1);
    assert.equal(env["MOM_DEV_HOST"], "1");
  });

  test("Linux host defaults to MoM and treats ordinary arguments as MoM arguments", () => {
    for (const args of [[], ["mine", "config.json"], ["--", "--host", "value"]]) {
      runDev("linux", "x64", {MOM_DEV_HOST: "1"}, args, (_command, forwarded) => {
        assert.equal(forwarded[0], path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"));
        assert.deepEqual(forwarded.slice(1), ["node", "mom.js", ...args]);
        return {status: 0};
      });
    }
  });

  test("host flags and separators are literal MoM arguments, not wrapper options", () => {
    for (const args of [["--host", "value"], ["--", "--host", "value"], ["--unknown", "value"]]) {
      runDev("linux", "x64", {}, args, (_command, forwarded) => {
        assert.equal(forwarded[0], path.resolve(__dirname, "../scripts/dev-container.sh"));
        assert.deepEqual(forwarded.slice(1), ["node", "mom.js", ...args]);
        return {status: 0};
      });
    }
    runDev("linux", "x64", {}, ["node", "mom.js", "--host", "--", "--exec"], (_command, forwarded) => {
      assert.deepEqual(forwarded.slice(1), ["node", "mom.js", "--host", "--", "--exec"]);
      return {status: 0};
    });
  });

  test("npm configuration is not an undocumented host selection shortcut", () => {
    for (const value of ["true", "false", "", "example.invalid"]) {
      const env = {npm_config_host: value};
      runDev("linux", "x64", env, ["node", "mom.js"], (_command, forwarded, options) => {
        assert.equal(forwarded[0], path.resolve(__dirname, "../scripts/dev-container.sh"));
        assert.equal(options.env, env);
        return {status: 0};
      });
    }
  });

  test("Linux host shares backend validation and vendor narrowing with Windows", () => {
    /** @type {NodeJS.Platform[]} */
    const platforms = ["linux", "win32"];
    for (const platform of platforms) {
      runDev(platform, "x64", {MOM_DEV_HOST: "1", MOM_GPU_BACKEND: "all"}, [], (_command, _args, options) => {
        assert.equal(options.env?.["MOM_GPU_BACKEND"], "all");
        assert.equal(options.env?.["MOM_GPU_TEST_VENDORS"], "intel,nvidia,amd");
        return {status: 0};
      });
    }
    assert.equal(nativeDevEnvironment({})["MOM_GPU_BACKEND"], "intel");
    assert.equal(nativeDevEnvironment({MOM_GPU_BACKEND: "amd", MOM_GPU_TEST_VENDORS: "custom"})["MOM_GPU_TEST_VENDORS"], "custom");
  });

  test("Linux host rejects unsupported architectures and invalid policy before spawning", () => {
    const spawn = () => {assert.fail("invalid native policy must not start a process");};
    assert.throws(() => runDev("linux", "arm64", {MOM_DEV_HOST: "1"}, [], spawn), /x86-64/);
    assert.throws(() => runDev("linux", "x64", {MOM_DEV_HOST: "1", MOM_GPU_BACKEND: "invalid"}, [], spawn), /Unknown MOM_GPU_BACKEND/);
    assert.throws(() => runDev("linux", "x64", {MOM_DEV_HOST: "1", MOM_REUSE_BUILT_WORKER: "yes"}, [], spawn), /must be 0 or 1/);
    assert.throws(() => runDev("linux", "x64", {MOM_DEV_HOST: "1", MOM_REUSE_BUILT_WORKER: "1"}, [], spawn),
      /unsupported in native development.*validates incremental caches/);
  });

  test("Linux host preserves source-entrypoint failures without separately launching the command", () => {
    for (const status of [29, null]) {
      let calls = 0;
      assert.equal(runDev("linux", "x64", {MOM_DEV_HOST: "1"}, ["node", "mom.js"], () => {
        calls++;
        return {status};
      }), status ?? 1);
      assert.equal(calls, 1);
    }
    assert.throws(() => runDev("linux", "x64", {MOM_DEV_HOST: "1"}, [], () => ({status: null, error: new Error("ENOENT")})), /ENOENT/);
  });

  test("Linux native build removes npm PATH scaffolding while preserving policy and caller environment", () => {
    const root = "/fixture/mom";
    const scaffold = [root + "/node_modules/.bin", "/fixture/node_modules/.bin", "/node_modules/.bin",
      "/fixture/npm/node_modules/@npmcli/run-script/lib/node-gyp-bin"];
    const env = {npm_execpath: "/fixture/npm/bin/npm-cli.js", npm_package_json: root + "/package.json",
      npm_lifecycle_event: "dev", PATH: [...scaffold, "/caller/tools"].join(":"), MOM_DEV_HOST: "1",
      MOM_LINUX_BUILD_COMPILER: "dpcpp-opencl"};
    runDev("linux", "x64", env, [], (_command, _args, options) => {
      assert.equal(options.env?.["PATH"], "/caller/tools");
      assert.equal(options.env?.["npm_package_json"], undefined);
      assert.equal(options.env?.["MOM_DEV_HOST"], "1");
      assert.equal(options.env?.["MOM_LINUX_BUILD_COMPILER"], "dpcpp-opencl");
      return {status: 0};
    });
    assert.equal(env.PATH, [...scaffold, "/caller/tools"].join(":"));
    assert.equal(env.npm_package_json, root + "/package.json");
  });

  test("Windows always builds natively regardless of the accepted host marker", () => {
    for (const marker of [undefined, "", "0", "1"]) {
      let calls = 0;
      runDev("win32", "x64", {MOM_DEV_HOST: marker}, ["mine", "config.json", "--host"], (command, args) => {
        if (calls++ === 0) {assert.equal(command, "powershell.exe");} else {
          assert.equal(command, process.execPath);
          assert.deepEqual(args, ["mom.js", "mine", "config.json", "--host"]);
        }
        return {status: 0};
      });
      assert.equal(calls, 2);
    }
  });

  test("Linux host selection accepts only the environment marker", () => {
    for (const marker of [undefined, "", "0", "1"]) {
      runDev("linux", "x64", {MOM_DEV_HOST: marker}, [], (_command, args) => {
        assert.equal(args[0], path.resolve(__dirname, "../scripts",
          marker === "1" ? "multicompiler-entrypoint.sh" : "dev-container.sh"));
        return {status: 0};
      });
    }
    /** @type {NodeJS.Platform[]} */
    const platforms = ["linux", "win32"];
    for (const platform of platforms) {
      for (const marker of ["yes", "true", "2", " 1", "1 "]) {
        assert.throws(() => runDev(platform, "x64", {MOM_DEV_HOST: marker}, [], () => {
          assert.fail("invalid host marker must not start a process");
        }), /MOM_DEV_HOST.*0 or 1/);
      }
    }
  });

  test("explicit Node/npm commands remain commands and generic commands require leading --exec", () => {
    for (const args of [["node", "mom.js", "--exec"], ["node.exe", "mom.js"], ["npm", "test", "--", "cpu"],
      ["npm.cmd", "run", "check"], ["--exec", "custom tool", "--", "a&b", "--host"]]) {
      runDev("linux", "x64", {}, args, (_command, forwarded, options) => {
        assert.deepEqual(forwarded.slice(1), args[0] === "--exec" ? args.slice(1) : args);
        assert.equal(options.shell, undefined);
        return {status: 0};
      });
    }
    runDev("linux", "x64", {}, ["custom-tool", "value"], (_command, args) => {
      assert.deepEqual(args.slice(1), ["node", "mom.js", "custom-tool", "value"]);
      return {status: 0};
    });
    /** @type {NodeJS.Platform[]} */
    const platforms = ["linux", "win32"];
    for (const platform of platforms) {
      for (const args of [["--exec"], ["--exec", ""]]) {
        assert.throws(() => runDev(platform, "x64", {}, args, () => {
          assert.fail("missing explicit command must not start a process");
        }), /command/i);
      }
    }
  });

  test("public npm entrypoint forwards host defaults and shell metacharacters without executing a builder", {
    skip: process.platform !== "linux", timeout: 30000,
  }, () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dev-public-npm-"));
    const preload = path.join(directory, "capture.js");
    const repo = path.resolve(__dirname, "..");
    // Intercept only the dispatcher process: npm itself retains its real argv quoting boundary.
    fs.writeFileSync(preload, `"use strict";
if (process.argv[1] === ${JSON.stringify(path.join(repo, "scripts", "dev.js"))}) {
  require("node:child_process").spawnSync = (command, args, options) => {
    console.log("MOM_DEV_CAPTURE=" + JSON.stringify({command, args, host: options.env.MOM_DEV_HOST || null,
      backend: options.env.MOM_GPU_BACKEND, shell: options.shell || null}));
    return {status: 0};
  };
}
`);
    try {
      const literal = '{"path":"a b","meta":"& % ! $ ` ; >"}';
      for (const {args, host, forwarded} of [
        {args: [], host: null, forwarded: ["node", "mom.js"]},
        {args: ["--", "mine", "config.json"], host: null, forwarded: ["node", "mom.js", "mine", "config.json"]},
        {args: ["--", "mine", "config.json", literal], host: "1", forwarded: ["node", "mom.js", "mine", "config.json", literal]},
        {args: [], host: "1", forwarded: ["node", "mom.js"]},
        {args: ["--", "--host", "--", literal], host: null, forwarded: ["node", "mom.js", "--host", "--", literal]},
        {args: ["--", "node", "mom.js", literal], host: "1", forwarded: ["node", "mom.js", literal]},
        {args: ["--", "npm", "test", "--", "cpu", literal], host: null, forwarded: ["npm", "test", "--", "cpu", literal]},
        {args: ["--", "--exec", "custom tool", literal], host: "1", forwarded: ["custom tool", literal]},
      ]) {
        const result = spawnSync("npm", ["run", "dev", ...args], {cwd: repo, encoding: "utf8", timeout: 10000,
          env: {...process.env, NODE_OPTIONS: `--require=${preload}`, MOM_GPU_BACKEND: "amd", MOM_REUSE_BUILT_WORKER: "0",
            MOM_DEV_HOST: host || ""}});
        assert.equal(result.error, undefined);
        assert.equal(result.status, 0);
        assert.equal(/Unknown cli config/.test(result.stderr), false);
        const rows = result.stdout.split(/\r?\n/).filter(line => line.startsWith("MOM_DEV_CAPTURE="));
        assert.equal(rows.length, 1);
        assert.ok(rows[0]);
        const capture = JSON.parse(rows[0].slice("MOM_DEV_CAPTURE=".length));
        assert.equal(capture.command, "bash");
        assert.equal(capture.host, host);
        assert.equal(capture.backend, "amd");
        assert.equal(capture.shell, null);
        assert.ok(capture.args[0].endsWith(host ? "scripts/multicompiler-entrypoint.sh" : "scripts/dev-container.sh"));
        assert.deepEqual(capture.args.slice(1), forwarded);
      }
    } finally {
      fs.rmSync(directory, {recursive: true, force: true});
    }
  });
});
