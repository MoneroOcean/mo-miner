"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const {spawnSync} = require("node:child_process");
const path = require("node:path");
const os = require("node:os");
const test = require("node:test");
const {buildRelease, releaseBuildCommands, releaseBuildEnvironment} = require("../scripts/build-release");
const {normalizeBuildEnvironment} = require("../scripts/build-environment");

test.describe("Native release build", {concurrency: false}, () => {
  test("npm development and deployment entrypoints use native platform dispatchers", () => {
    assert.equal(require("../package.json").scripts.dev, "node scripts/dev.js");
    assert.equal(require("../package.json").scripts["test:deploy"], "node scripts/test-deploy.js");
    assert.equal(require("../package.json").scripts["build:release"], "node scripts/build-release.js");
  });

  test("native release command selects existing producers without Docker", () => {
    const linux = releaseBuildCommands("linux");
    const windows = releaseBuildCommands("win32");
    assert.equal(linux.length, 1);
    assert.equal(windows.length, 2);
    assert.equal(linux[0]?.[0], "bash");
    assert.deepEqual(linux[0]?.slice(1, 3), ["-eu", "-c"]);
    assert.equal(linux[0]?.[4], "--");
    assert.ok(linux[0]?.[5]?.endsWith("multicompiler-entrypoint.sh"));
    assert.ok(linux[0]?.[6]?.endsWith("package-linux-combined.sh"));
    assert.ok(windows[0]?.includes("-Backend"));
    assert.ok(windows[0]?.includes("all"));
    assert.ok(windows[1]?.includes("-Version"));
    for (const command of [...linux, ...windows]) {
      assert.equal(command.some(arg => /docker|scripts\/dev-container\.sh$/.test(arg)), false);
    }
  });

  test("actual npm release invocation repeats the child environment and stops before packaging on failure", () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-npm-release-test-"));
    try {
      fs.mkdirSync(path.join(temp, "scripts"));
      const metadata = require("../package.json");
      fs.writeFileSync(path.join(temp, "package.json"), JSON.stringify({version: metadata.version,
        scripts: {"build:release": metadata.scripts["build:release"], nested: "node scripts/nested.cjs"}}));
      fs.copyFileSync(path.join(__dirname, "../scripts/build-release.js"), path.join(temp, "scripts/build-release.js"));
      fs.copyFileSync(path.join(__dirname, "../scripts/build-environment.js"), path.join(temp, "scripts/build-environment.js"));
      fs.writeFileSync(path.join(temp, "scripts/nested.cjs"), [
        '"use strict";',
        'const {spawn} = require("node:child_process");',
        'const {normalizeBuildEnvironment} = require("./build-environment");',
        'const child = spawn(process.execPath, [process.env.npm_execpath, "run", "build:release"],',
        '  {env: normalizeBuildEnvironment(process.env), stdio: "inherit"});',
        'child.on("error", () => {process.exitCode = 1;});',
        'child.on("exit", code => {process.exitCode = code ?? 1;});',
      ].join("\n"));
      const preload = path.join(temp, "capture.cjs");
      // Intercept the real CLI's builder boundary; npm and the release CLI still run normally.
      fs.writeFileSync(preload, [
        '"use strict";',
        'require("node:child_process").spawnSync = (command, args, options) => {',
        '  console.log("FIXTURE_RELEASE=" + JSON.stringify({command, args, cwd: options.cwd,',
        "    path: options.env[Object.keys(options.env).find(key => /^path$/i.test(key))],",
        "    force: options.env.MOM_FORCE_REBUILD,",
        "    backend: options.env.MOM_GPU_BACKEND, portable: options.env.MOM_PORTABLE_BUILD,",
        "    lifecycle: options.env.npm_lifecycle_event}));",
        "  return {status: Number(process.env.FIXTURE_RELEASE_STATUS || 0)};",
        "};",
      ].join("\n"));
      const npmCli = process.env["npm_execpath"];
      const command = npmCli ? process.execPath : "npm";
      const npmArgs = [...(npmCli ? [npmCli] : []), "run", "build:release"];
      const baseEnv = normalizeBuildEnvironment(process.env);
      if (process.platform === "win32") {
        const pathKeys = Object.keys(baseEnv).filter(key => /^path$/i.test(key));
        const pathValue = baseEnv[pathKeys[0] || "PATH"] || "";
        for (const key of pathKeys) {delete baseEnv[key];}
        baseEnv["PATH"] = pathValue;
      }
      /** @type {NodeJS.ProcessEnv} */
      const childEnv = {...baseEnv, NODE_OPTIONS: `--require=${JSON.stringify(preload)}`,
        MOM_FORCE_REBUILD: "0", MOM_CPU_MARCH: "", MOM_LINUX_BUILD_PLAN_ONLY: "0",
        MOM_DPCPP_IMPL: "", MOM_COMBINED_TARGETS: "", FIXTURE_RELEASE_STATUS: "0",
        npm_config_update_notifier: "false", npm_config_audit: "false", npm_config_fund: "false"};
      const direct = spawnSync(process.execPath, ["scripts/build-release.js"], {
        cwd: temp, encoding: "utf8", timeout: 20000, env: childEnv});
      assert.equal(direct.status, 0);
      const directPaths = direct.stdout.split(/\r?\n/).filter(line => line.startsWith("FIXTURE_RELEASE="))
        .map(line => JSON.parse(line.slice("FIXTURE_RELEASE=".length)).path);
      /** @type {{command: string, args: string[], cwd: string, path: string, force: string,
       * backend: string, portable: string, lifecycle: string}[][]} */
      const repeated = [];
      for (const status of [0, 0, 37]) {
        const result = spawnSync(command, npmArgs, {cwd: temp, encoding: "utf8", timeout: 20000,
          shell: !npmCli && process.platform === "win32",
          env: {...childEnv, FIXTURE_RELEASE_STATUS: String(status)}});
        assert.equal(result.error, undefined);
        assert.equal(result.status, status);
        const calls = result.stdout.split(/\r?\n/).filter(line => line.startsWith("FIXTURE_RELEASE="))
          .map(line => JSON.parse(line.slice("FIXTURE_RELEASE=".length)));
        assert.equal(calls.length, process.platform === "linux" || status !== 0 ? 1 : 2);
        for (const call of calls) {
          assert.equal(call.cwd, temp);
          assert.equal(call.force, "0");
          assert.equal(call.backend, "all");
          assert.equal(call.portable, "1");
          assert.equal(call.lifecycle, "build:release");
          assert.equal(typeof call.path, "string");
        }
        assert.deepEqual(calls.map(call => call.path), directPaths.slice(0, calls.length));
        repeated.push(calls);
      }
      assert.deepEqual(repeated[0], repeated[1]);
      assert.deepEqual(repeated[2], repeated[0]?.slice(0, 1));
      // A deliberate caller .bin path is not npm scaffolding and remains compiler context.
      const callerPath = path.join(temp, "node_modules/.bin") + path.delimiter + (childEnv["PATH"] || "");
      const changed = spawnSync(command, npmArgs, {cwd: temp, encoding: "utf8", timeout: 20000,
        shell: !npmCli && process.platform === "win32", env: {...childEnv, PATH: callerPath}});
      assert.equal(changed.status, 0);
      const changedPaths = changed.stdout.split(/\r?\n/).filter(line => line.startsWith("FIXTURE_RELEASE="))
        .map(line => JSON.parse(line.slice("FIXTURE_RELEASE=".length)).path);
      assert.deepEqual(changedPaths, directPaths.map(() => callerPath));
      assert.notDeepEqual(changedPaths, directPaths);
      const nested = spawnSync(command, [...(npmCli ? [npmCli] : []), "run", "nested"], {
        cwd: temp, encoding: "utf8", timeout: 20000,
        shell: !npmCli && process.platform === "win32", env: childEnv});
      assert.equal(nested.error, undefined);
      assert.equal(nested.status, 0);
      const nestedCalls = nested.stdout.split(/\r?\n/).filter(line => line.startsWith("FIXTURE_RELEASE="))
        .map(line => JSON.parse(line.slice("FIXTURE_RELEASE=".length)));
      assert.deepEqual(nestedCalls, repeated[0]);
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  });

  test("release builds override targeted workers without mutating caller environment", () => {
    const original = {MOM_GPU_BACKEND: "intel", MOM_LINUX_BUILD_COMPILER: "oneapi",
      MOM_REUSE_BUILT_WORKER: "1", MOM_BUILD_JOBS: "2", PATH: "caller-selected-tools"};
    const env = releaseBuildEnvironment(original);
    assert.equal(original.MOM_GPU_BACKEND, "intel");
    assert.equal(env["MOM_GPU_BACKEND"], "all");
    assert.equal(env["MOM_LINUX_BUILD_COMPILER"], "all");
    assert.equal(env["MOM_PORTABLE_BUILD"], "1");
    assert.equal(env["MOM_REUSE_BUILT_WORKER"], "0");
    assert.equal(env["MOM_BUILD_JOBS"], "2");
    assert.equal(env["PATH"], original.PATH);
    assert.throws(() => releaseBuildEnvironment({MOM_CPU_MARCH: "native"}), /Unset MOM_CPU_MARCH/);
  });

  test("release builds preserve explicit clean requests on both platforms", () => {
    for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
      const options = {MOM_FORCE_REBUILD: "1"};
      let calls = 0;
      assert.equal(buildRelease(platform, "x64", options, false, (_command, _args, invocation) => {
        assert.equal(invocation.env?.["MOM_FORCE_REBUILD"], "1");
        calls++;
        return {status: 0};
      }), 0);
      assert.equal(calls, platform === "linux" ? 1 : 2);
      assert.deepEqual(options, {MOM_FORCE_REBUILD: "1"});
    }
    for (const value of ["true", "yes", "2"]) {
      assert.throws(() => releaseBuildEnvironment({MOM_FORCE_REBUILD: value}), /MOM_FORCE_REBUILD/);
    }
  });

  test("release CLI rejects contradictory clean and package-only modes before building", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/build-release.js"),
      "--clean", "--package-only"], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Usage:.*--clean/);
    assert.equal(result.stdout, "");
  });

  test("native release lifecycle never packages a failed build", () => {
  /** @type {{command: string, args: string[], cwd: import("node:child_process").SpawnSyncOptions["cwd"]}[]} */
    const commands = [];
    /** @type {import("../scripts/build-release").ReleaseSpawn} */
    const spawn = (command, args, options) => {
      commands.push({command, args, cwd: options.cwd});
      return {status: 7};
    };
    assert.equal(buildRelease("linux", "x64", {}, false, spawn), 7);
    assert.equal(commands.length, 1);
    assert.equal(commands[0]?.cwd, path.resolve(__dirname, ".."));
  });

  test("Linux release owns the shared lock across build and packaging, including package-only", {
    skip: process.platform !== "linux",
  }, () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-lock space-"));
    try {
      assert.equal(spawnSync("git", ["init", "--quiet", temp]).status, 0);
      const lockDir = path.join(temp, ".git", "mom-locks");
      const lock = path.join(lockDir, `mom-r-sh-${process.getuid?.()}.build.lock`);
      fs.mkdirSync(lockDir);
      fs.writeFileSync(lock, "");
      const builder = path.join(temp, "builder space.sh");
      const packager = path.join(temp, "packager space.sh");
      const checkLock = 'if [ "$FIXTURE_REQUIRE_LOCK" = 1 ] && flock -n "$FIXTURE_LOCK" true; then exit 91; fi\n';
      fs.writeFileSync(builder, "set -eu\n" + checkLock +
        // eslint-disable-next-line no-template-curly-in-string -- Bash parameter expansion.
        '[ "${MOM_NATIVE_BUILD_LOCK:-0}" = 1 ] || [ "${MOM_R_SH:-0}" = 1 ] || exit 92\nprintf "build\\n" >> "$FIXTURE_EVENTS"\nexit "$FIXTURE_BUILD_STATUS"\n');
      fs.writeFileSync(packager, "set -eu\n" + checkLock +
        '[ "$1" = 9.8.7 ] || exit 93\nprintf "package\\n" >> "$FIXTURE_EVENTS"\n');
      const events = path.join(temp, "events");
      for (const owner of ["release", "container", "native-dev"]) {
        for (const packageOnly of [false, true]) {
          for (const status of [0, 37]) {
            fs.writeFileSync(events, "");
            let calls = 0;
            const result = buildRelease("linux", "x64", {MOM_RELEASE_VERSION: "9.8.7",
              MOM_DEV_HOST: "1", MOM_R_SH: owner === "container" ? "1" : "0",
              MOM_NATIVE_BUILD_LOCK: owner === "native-dev" ? "1" : "0"}, packageOnly,
            (command, args, options) => {
              calls++;
              // Exercise the actual lifecycle shell, substituting only its producer paths.
              const forwarded = args.map(arg => arg.endsWith("multicompiler-entrypoint.sh") ? builder :
                arg.endsWith("package-linux-combined.sh") ? packager : arg);
              // Stand in for the development parent's already-held lock without Docker.
              const child = spawnSync(owner === "release" ? command : "flock",
                owner === "release" ? forwarded : [lock, command, ...forwarded],
                {cwd: temp, encoding: "utf8", timeout: 5000,
                  env: {...process.env, ...options.env, FIXTURE_LOCK: lock, FIXTURE_EVENTS: events,
                    FIXTURE_REQUIRE_LOCK: "1", FIXTURE_BUILD_STATUS: String(status)}});
              assert.equal(child.error, undefined);
              return child;
            });
            assert.equal(result, packageOnly ? 0 : status);
            assert.equal(calls, 1);
            assert.equal(fs.readFileSync(events, "utf8"), packageOnly ? "package\n" :
              status === 0 ? "build\npackage\n" : "build\n");
            assert.equal(spawnSync("flock", ["-n", lock, "true"]).status, 0);
          }
        }
      }
      assert.ok(fs.existsSync(lockDir));
      const entrypoint = fs.readFileSync(path.join(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
      assert.match(entrypoint, /if \[ "\$\{MOM_DEV_HOST:-0\}" = 1 \] && \[ "\$\{MOM_NATIVE_BUILD_LOCK:-0\}" != 1 \] && \[ "\$\{MOM_R_SH:-0\}" != 1 \]; then/);
      assert.match(entrypoint, /flock 9\n {2}export MOM_NATIVE_BUILD_LOCK=1\nfi/);
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  });

  test("release builds reject inherited controls that skip or change policy workers", () => {
    for (const [name, value] of Object.entries({
      MOM_LINUX_BUILD_PLAN_ONLY: "1",
      MOM_DPCPP_IMPL: "dpcpp-opencl",
      MOM_COMBINED_TARGETS: "spir64",
    })) {
      let calls = 0;
      assert.throws(() => buildRelease("linux", "x64", {[name]: value}, false,
        () => { calls++; return {status: 0}; }), new RegExp(`Unset ${name}`));
      assert.equal(calls, 0);
    }
    assert.doesNotThrow(() => releaseBuildEnvironment({MOM_LINUX_BUILD_PLAN_ONLY: "0",
      MOM_DPCPP_IMPL: "", MOM_COMBINED_TARGETS: ""}));
  });

  test("native release lifecycle propagates spawn failure and rejects unsupported hosts", () => {
    const spawn = () => ({status: null, error: new Error("missing build shell")});
    assert.throws(() => buildRelease("linux", "x64", {}, false, spawn), /missing build shell/);
    assert.throws(() => buildRelease("darwin", "x64", {}), /Linux and Windows/);
    assert.throws(() => buildRelease("linux", "arm64", {}), /x86-64/);
  });

  test("CI packaging and native builds share the same producer", () => {
    for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
      if (platform === "linux") {
        assert.deepEqual(releaseBuildCommands(platform, true)[0]?.slice(0, -1), releaseBuildCommands(platform)[0]?.slice(0, -1));
        assert.equal(releaseBuildCommands(platform, true)[0]?.at(-1), "1");
      } else {
        assert.deepEqual(releaseBuildCommands(platform, true), releaseBuildCommands(platform).slice(1));
      }
      let calls = 0;
      assert.equal(buildRelease(platform, "x64", {}, true, () => { calls++; return {status: 0}; }), 0);
      assert.equal(calls, 1);
    }
  });

  test("release overrides normalize versions and reject unsafe values before launch", () => {
    for (const platform of /** @type {NodeJS.Platform[]} */ (["linux", "win32"])) {
      for (const version of ["9.8.7", "v9.8.7", "V9.8.7"]) {
        /** @type {string[][]} */
        const calls = [];
        assert.equal(buildRelease(platform, "x64", {MOM_RELEASE_VERSION: version}, false,
          (command, args) => { calls.push([command, ...args]); return {status: 0}; }), 0);
        assert.equal(calls.length, platform === "linux" ? 1 : 2);
        assert.equal(platform === "linux" ? calls[0]?.at(-2) : calls[1]?.at(-1), "9.8.7");
      }
      for (const version of ["../9", "v", "9/8", "9'; exit 0", "9\n8", "9\n"]) {
        let calls = 0;
        assert.throws(() => buildRelease(platform, "x64", {MOM_RELEASE_VERSION: version}, false,
          () => { calls++; return {status: 0}; }), /Invalid release version/);
        assert.equal(calls, 0);
      }
    }
  });

  test("Windows deployment uses the native release lifecycle inside the existing VM", {
    skip: process.platform === "win32",
  }, () => {
    const deploy = fs.readFileSync(path.join(__dirname, "../scripts/test-deploy.sh"), "utf8");
    const build = deploy.match(/^package_windows_release\(\) \{\n[\s\S]*?\n\}\n/m)?.[0];
    assert.ok(build);
    for (const status of [0, 37]) {
      /** @type {import("node:child_process").SpawnSyncReturns<string>} */
      const result = spawnSync("bash", ["-c", [
        "windows_available() { return 0; }",
        'run_windows_root() { printf "%s\\n" "$@"; return "$FIXTURE_STATUS"; }',
        build,
        "package_windows_release",
      ].join("\n")], {encoding: "utf8", env: {...process.env,
        MOM_DEPLOY_REUSE_ARCHIVE: "0", WINDOWS_ARCHIVE: "mom-v9.8.7-win.zip",
        MOM_FORCE_REBUILD: "1",
        RELEASE_VERSION: "9.8.7", WIN_RUN: "fixture-run", WIN_MOM_DEV_BASE: "fixture-base",
        FIXTURE_STATUS: String(status)}});
      assert.equal(result.status, status, result.stdout + result.stderr);
      assert.match(result.stdout, /GPU_GROUP=none\nWIN_MOM_RUN_BASE=fixture-base\nfixture-run\n--release\n/);
      assert.match(result.stdout, /--download\nbuild\/win\n--download\nmom-v9.8.7-win.zip\n--\nenv\nMOM_RELEASE_VERSION=9.8.7\nMOM_FORCE_REBUILD=1\n/);
      const command = result.stdout.split("\n").find(line => line.startsWith("npm install"));
      assert.ok(command);
      assert.equal(command.match(/npm run build:release/g)?.length, 1);
      assert.doesNotMatch(command, /--package-only|build-windows|package-windows/);
      // The single elevated wrapper retains credentials through build completion; actual
      // dispatcher regressions verify failure and missing-archive branches before chown.
      assert.match(result.stdout, /"\$@" \|\| exit/);
      assert.match(result.stdout, /\[ -f "\$archive" \]/);
      assert.match(result.stdout, /chown -R "\$owner" build\/win "\$archive"/);
    }
    for (const status of [0, 37]) {
      /** @type {string[][]} */
      const calls = [];
      assert.equal(buildRelease("win32", "x64", {}, false, (command, args) => {
        calls.push([command, ...args]);
        return {status: calls.length === 1 ? status : 0};
      }), status);
      assert.equal(calls.filter(command => command.some(arg => arg.endsWith("build-windows-multicompiler.ps1"))).length, 1);
      assert.equal(calls.length, status === 0 ? 2 : 1);
    }
  });
});
