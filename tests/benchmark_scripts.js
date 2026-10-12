"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {runInNewContext} = require("node:vm");
const benchmark = require("../scripts/benchmark-gpu-algos");
const release = require("../scripts/check-release-performance");
const comparison = require("../scripts/compare-gpu-benchmarks");
const performance = require("../scripts/readme-performance");

test.describe("Correctness test dispatcher", {concurrency: false}, () => {
  const source = fs.readFileSync(path.join(__dirname, "run_hash.js"), "utf8");
  const logicFiles = [
    "tests/logic.js", "tests/compiler_policy.js", "tests/portable_dag_modes.js",
    "tests/nvidia_compatibility.js", "tests/intel_device_selection.js", "tests/gpu_test_matrix_order.js",
    "tests/pearlhash_f16_lowering.js", "tests/readme_performance.js",
    "tests/benchmark_scripts.js", "tests/windows_command.js", "tests/pool_transport.js",
    "tests/helper_workers.js", "tests/release_archive.js", "tests/build_release.js", "tests/build_environment.js",
    "tests/incremental_linux_build.js", "tests/windows_incremental_build.js",
    "tests/windows_incremental_orchestration.js",
    "tests/windows_worker_discovery.js", "tests/dev.js", "tests/native_deploy.js",
  ];

  /** @param {string | undefined} suite
   * @param {{sourceCheckout?: boolean, platform?: string, env?: NodeJS.ProcessEnv}} [options] */
  function dispatch(suite, {sourceCheckout = true, platform = "linux", env = {}} = {}) {
    /** @type {Array<{command: string, args: string[], env: NodeJS.ProcessEnv}>} */
    const calls = [];
    /** @type {Array<{args: string[], env: NodeJS.ProcessEnv}>} */
    const routed = [];
    /** @type {string[]} */
    const errors = [];
    /** @type {number | undefined} */
    let exitCode;
    const stopped = new Error("fixture process exit");
    const minerCommand = {
      repoRoot: "/fixture/source",
      /** @param {string[]} args @param {NodeJS.ProcessEnv} runnerEnv */
      resolveNodeRunner(args, runnerEnv) {
        routed.push({args: Array.from(args), env: {...runnerEnv}});
        return {command: "fixture-hardware-runner", args, env: runnerEnv};
      },
      /** @param {string} command @param {string[]} args @param {{env: NodeJS.ProcessEnv}} options */
      spawnAndExit(command, args, options) {
        calls.push({command, args: Array.from(args), env: {...options.env}});
      },
    };
    try {
      runInNewContext(source, {
        /** @param {string} name */
        require(name) {
          if (name === "node:fs") {return {existsSync: () => sourceCheckout};}
          if (name === "node:path") {return path;}
          if (name === "./common/miner_command") {return minerCommand;}
          throw new Error(`Unexpected dispatcher dependency: ${name}`);
        },
        process: {
          argv: ["fixture-node", "tests/run_hash.js", ...(suite ? [suite] : [])],
          env, platform, execPath: "fixture-local-node",
          /** @param {number} code */
          exit(code) {exitCode = code; throw stopped;},
        },
        console: {
          /** @param {string} message */
          error(message) {errors.push(message);},
        },
      }, {filename: "tests/run_hash.js"});
    } catch (error) {
      if (error !== stopped) {throw error;}
    }
    return {calls, routed, errors, exitCode};
  }

  /** @param {{calls: Array<{args: string[]}>}} result */
  function dispatchedFiles(result) {
    assert.equal(result.calls.length, 1);
    const args = result.calls[0]?.args;
    assert.ok(args);
    assert.deepEqual(args.slice(0, 2), ["--require", "./tests/common/no_pool_network.js"]);
    assert.ok(args.includes("--test-concurrency=1"));
    return args.filter(arg => /^tests\/.*\.js$/.test(arg));
  }

  test("npm exposes only maintained commands with matching descriptions", () => {
    const metadata = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
    const expected = ["dev", "build:release", "check", "lint", "test", "test:perf",
      "test:deploy", "typecheck"].sort();
    assert.deepEqual(Object.keys(metadata.scripts).sort(), expected);
    assert.deepEqual(Object.keys(metadata.scriptDescriptions).sort(), expected);
    for (const description of Object.values(metadata.scriptDescriptions)) {
      assert.equal(typeof description, "string");
      assert.ok(String(description).trim());
    }
    assert.equal(metadata.scripts.test, "npm run check && node tests/run_hash.js");
  });

  test("npm test checks first, forwards the selector, and stops when checks fail", () => {
    const metadata = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-npm-test-dispatch-"));
    try {
      fs.mkdirSync(path.join(temp, "tests"));
      fs.writeFileSync(path.join(temp, "package.json"), JSON.stringify({scripts: {
        check: "node check.cjs", test: metadata.scripts.test,
      }}));
      fs.writeFileSync(path.join(temp, "check.cjs"),
        'console.log("FIXTURE_CHECK");process.exitCode=process.env.FIXTURE_CHECK_FAIL==="1"?7:0;\n');
      fs.writeFileSync(path.join(temp, "tests", "run_hash.js"),
        'console.log("FIXTURE_HASH="+JSON.stringify(process.argv.slice(2)));\n');
      const npmCli = process.env["npm_execpath"];
      const command = npmCli ? process.execPath : "npm";
      const args = [...(npmCli ? [npmCli] : []), "test", "--", "gpu-integrated"];
      for (const fail of [false, true]) {
        const result = spawnSync(command, args, {
          cwd: temp, encoding: "utf8", timeout: 20000,
          shell: !npmCli && process.platform === "win32",
          env: {...process.env, FIXTURE_CHECK_FAIL: fail ? "1" : "0",
            npm_config_update_notifier: "false", npm_config_audit: "false", npm_config_fund: "false"},
        });
        assert.equal(result.error, undefined);
        const markers = result.stdout.split(/\r?\n/).filter(line => line.startsWith("FIXTURE_"));
        assert.deepEqual(markers, fail ? ["FIXTURE_CHECK"]
          : ["FIXTURE_CHECK", 'FIXTURE_HASH=["gpu-integrated"]']);
        if (fail) {
          assert.notEqual(result.status, 0, result.stderr);
        } else {
          assert.equal(result.status, 0, result.stderr);
        }
      }
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  });

  test("logic runs every former host logic file once through local Node", () => {
    for (const platform of ["linux", "win32", "darwin"]) {
      const result = dispatch("logic", {platform});
      const files = dispatchedFiles(result);
      assert.deepEqual([...files].sort(), [...logicFiles].sort());
      assert.equal(new Set(files).size, files.length);
      assert.equal(result.calls[0]?.command, "fixture-local-node");
      assert.equal(result.routed.length, 0);
    }
  });

  test("default all shares canonical logic coverage and routes hardware", () => {
    const result = dispatch(undefined);
    const files = dispatchedFiles(result);
    assert.deepEqual(files.filter(file => logicFiles.includes(file)).sort(), [...logicFiles].sort());
    assert.ok(files.includes("tests/native_addon.js"));
    assert.ok(files.includes("tests/all.js"));
    assert.equal(new Set(files).size, files.length);
    assert.equal(result.routed.length, 1);
  });

  test("host runs CPU references and canonical logic locally", () => {
    const result = dispatch("host");
    const files = dispatchedFiles(result);
    assert.deepEqual([...files].sort(), ["tests/host.js", ...logicFiles,
      "tests/hoohash_host.js", "tests/pearlhash_claim.js"].sort());
    assert.equal(new Set(files).size, files.length);
    assert.equal(result.calls[0]?.command, "fixture-local-node");
    assert.equal(result.routed.length, 0);
    assert.equal(files.includes("tests/native_addon.js"), false);
  });

  for (const [suite, files] of Object.entries({
    cpu: ["tests/cpu.js", "tests/native_addon.js"],
    gpu: ["tests/gpu.js"],
    "gpu-discrete": ["tests/discrete_gpu.js"],
    "gpu-integrated": ["tests/integrated_gpu.js"],
    "gpu-multi": ["tests/multi_gpu.js"],
    "gpu-portable-cpu": ["tests/portable_gpu_cpu.js"],
    "gpu-fishhash-batch": ["tests/fishhash_batch.js"],
  })) {
    test(`${suite} retains hardware dispatch and preloads the pool guard`, () => {
      const result = dispatch(suite);
      assert.deepEqual(dispatchedFiles(result), files);
      assert.equal(result.routed.length, 1);
      assert.equal(result.calls[0]?.command, "fixture-hardware-runner");
    });
  }

  for (const suite of ["logic", "host"]) {
    test(`${suite} rejects a release-only checkout before dispatch`, () => {
      const result = dispatch(suite, {sourceCheckout: false});
      assert.equal(result.exitCode, 1);
      assert.ok(result.errors.length);
      assert.equal(result.calls.length, 0);
      assert.equal(result.routed.length, 0);
    });
  }

  test("host rejects non-Linux platforms without dispatching references", () => {
    for (const platform of ["win32", "darwin"]) {
      const result = dispatch("host", {platform});
      assert.equal(result.exitCode, 1);
      assert.ok(result.errors.length);
      assert.equal(result.calls.length, 0);
      assert.equal(result.routed.length, 0);
    }
  });

  test("release default omits source-only files but keeps guarded hash testing", () => {
    const result = dispatch(undefined, {sourceCheckout: false});
    assert.deepEqual(dispatchedFiles(result), ["tests/all.js"]);
    assert.equal(result.routed.length, 1);
  });

  test("unknown suite fails before runner resolution", () => {
    for (const suite of ["not-a-suite", "constructor", "__proto__"]) {
      const result = dispatch(suite);
      assert.equal(result.exitCode, 1);
      assert.ok(result.errors.length);
      assert.equal(result.calls.length, 0);
      assert.equal(result.routed.length, 0);
    }
  });

  test("generic OpenCL matrix clears inherited vendor labels while native routing preserves them", () => {
    for (const suite of ["all", "gpu", "gpu-discrete"]) {
      const opencl = dispatch(suite, {env: {MOM_GPU_BACKEND: "opencl", MOM_GPU_TEST_VENDORS: "intel"}});
      assert.deepEqual(opencl.calls[0]?.env, {MOM_GPU_TEST_VENDORS: undefined});
      const native = dispatch(suite, {env: {MOM_GPU_BACKEND: "intel", MOM_GPU_TEST_VENDORS: "intel"}});
      assert.deepEqual(native.calls[0]?.env, {MOM_GPU_TEST_VENDORS: "intel"});
    }
  });
});

test.describe("Deployment dispatcher failure boundaries", {concurrency: false,
  skip: process.platform !== "linux"}, () => {
  /** Run the unchanged entrypoints; only host tools and the VM boundary are fixtures.
   * @param {NodeJS.ProcessEnv} [controls] */
  function deploy(controls = {}) {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-dispatch-"));
    const bin = path.join(temp, "bin");
    fs.mkdirSync(bin);
    fs.mkdirSync(path.join(temp, "scripts"));
    const drm = path.join(temp, "drm");
    const dri = path.join(temp, "dri");
    fs.mkdirSync(path.join(drm, "card1/device/drm/renderD129"), {recursive: true});
    fs.writeFileSync(path.join(drm, "card1/device/vendor"), "0x1002");
    fs.mkdirSync(dri);
    fs.symlinkSync("/dev/null", path.join(dri, "renderD129"));
    fs.writeFileSync(path.join(temp, "kfd"), "fixture device presence");
    for (const name of ["test-deploy.sh", "dev-container.sh"]) {
      const destination = path.join(temp, "scripts", name);
      const source = fs.readFileSync(path.join(__dirname, "../scripts", name), "utf8");
      fs.writeFileSync(destination, name === "test-deploy.sh" ? source.replaceAll("/sys/class/drm", drm)
        .replaceAll("/dev/dri", dri).replaceAll("/dev/kfd", path.join(temp, "kfd"))
        .replace("INTERRUPT_GRACE_SECONDS=60", () => controls["FIXTURE_SHORT_INTERRUPT_GRACE"] === "1"
          ? "INTERRUPT_GRACE_SECONDS=1" : "INTERRUPT_GRACE_SECONDS=60") : source);
      fs.chmodSync(destination, 0o755);
    }
    fs.writeFileSync(path.join(temp, "package.json"), JSON.stringify({version: "9.8.7"}));
    const eventsPath = path.join(temp, "events.jsonl");
    const helpersPath = path.join(temp, "helpers.jsonl");
    const credentialPath = path.join(temp, "credential");
    const refreshRequestPath = path.join(temp, "refresh-request");
    const refreshFailurePath = path.join(temp, "refresh-failed");
    // Events contain only fixed labels/booleans: never capture sudo stdin or credentials.
    const stub = [
      `#!${process.execPath}`,
      'const fs = require("node:fs"), path = require("node:path"), cp = require("node:child_process");',
      "const tool = path.basename(process.argv[1]), args = process.argv.slice(2), env = process.env;",
      'const event = (kind, extra = {}) => fs.appendFileSync(env.FIXTURE_EVENTS, JSON.stringify({kind, ...extra}) + "\\n");',
      // Atomic publication keeps concurrent credential readers away from transient empty files.
      'const publish = (file, value) => { const temporary = file + "." + process.pid; fs.writeFileSync(temporary, String(value)); fs.renameSync(temporary, file); };',
      'const epoch = file => fs.existsSync(file) ? Number(fs.readFileSync(file, "utf8")) : 0;',
      'const exec = (command, argv, more = {}) => { const result = cp.spawnSync(command, argv, {env: {...env, ...more}, stdio: "inherit"}); process.exit(result.status ?? 99); };',
      'if (tool === "setsid") { event("active"); if (env.FIXTURE_REAL_SESSION === "1") exec("/usr/bin/setsid", args); exec(args[0], args.slice(1), {FIXTURE_ACTIVE: "1"}); }',
      'if (tool === "git") { console.log(path.join(env.FIXTURE_ROOT, "locks")); process.exit(0); }',
      'if (tool === "flock") process.exit(0);',
      'if (tool === "id") { console.log(env.FIXTURE_ROOT_USER === "1" ? "0" : "1000"); process.exit(0); }',
      'if (tool === "chown") { event("chown"); process.exit(0); }',
      'if (tool === "nvidia-smi") { console.log("GPU-12345678-1234-1234-1234-123456789abc"); process.exit(0); }',
      'if (tool === "sleep") {',
      '  if (args[0] !== "60") exec("/bin/sleep", args);',
      '  fs.appendFileSync(env.FIXTURE_HELPERS, JSON.stringify([process.pid, process.ppid]) + "\\n"); event("refresh-wait");',
      '  process.on("SIGTERM", () => { event("refresh-wait-stopped"); process.exit(0); });',
      '  process.on("SIGINT", () => { event("refresh-wait-stopped"); process.exit(0); });',
      '  if (env.FIXTURE_LONG_LINUX === "1") { const poll = setInterval(() => { if (epoch(env.FIXTURE_REFRESH_REQUEST) > epoch(env.FIXTURE_CREDENTIAL)) { clearInterval(poll); process.exit(0); } }, 10); setTimeout(() => { event("fixture-timeout"); process.exit(91); }, 10000); }',
      "  else setTimeout(() => process.exit(0), 10000);",
      "}",
      'if (tool === "cp") { event("copy", {exists: fs.existsSync(args[1])}); if (env.FIXTURE_COPY_FAIL === "1") process.exit(55); exec("/bin/cp", args); }',
      'if (tool === "sudo") {',
      '  const auth = args.includes("-v");',
      '  if (env.FIXTURE_REAL_SESSION === "1") { const groups = cp.spawnSync("/usr/bin/ps", ["-o", "sid=,pgid=", "-p", String(process.pid)], {encoding: "utf8"}).stdout.trim().split(/\\s+/); event("session", {sameSession: groups[0] === env.FIXTURE_EXPECTED_SID, sameCallerGroup: groups[1] === env.FIXTURE_EXPECTED_PGID, auth}); }',
      '  event(auth ? "auth" : "sudo", {active: env.FIXTURE_ACTIVE === "1", captured: fs.fstatSync(2).isFile(), noninteractive: args.includes("-n"), passwordInput: args.includes("-S")});',
      '  if (env.FIXTURE_AUTH_FAIL === "1") { console.error("sudo: fixture authentication required"); process.exit(1); }',
      '  if (auth && env.FIXTURE_REFRESH_FAIL === "1" && fs.existsSync(env.FIXTURE_CREDENTIAL)) { event("refresh-failed"); publish(env.FIXTURE_REFRESH_FAILURE, 1); process.exit(1); }',
      "  if (auth) { publish(env.FIXTURE_CREDENTIAL, epoch(env.FIXTURE_REFRESH_REQUEST)); process.exit(0); }",
      '  if (env.FIXTURE_LONG_LINUX === "1" && !args.includes("-S") && epoch(env.FIXTURE_CREDENTIAL) !== epoch(env.FIXTURE_REFRESH_REQUEST)) { event("credential-expired"); process.exit(1); }',
      '  const at = args.indexOf("--"); const command = args[at + 1];',
      '  if (command === "chown") process.exit(0);',
      "  if (!command) process.exit(98);",
      "  exec(command, args.slice(at + 2));",
      "}",
      'if (tool === "fixture-win") {',
      '  const build = args.includes("build/win"); event(build ? "windows-build" : "windows-lane", {multi: args.includes("gpu-multi"), discreteGroup: env.GPU_GROUP === "discrete"});',
      "  if (build && env.FIXTURE_SIGNAL) {",
      // The caller retains INT/TERM status, but Windows sudo/root cancellation always relays TERM.
      '    const callerSignal = "SIG" + env.FIXTURE_SIGNAL, signal = "SIGTERM";',
      '    const child = cp.spawn(process.execPath, ["-e", \'const fs=require("node:fs"); process.on(process.env.FIXTURE_SIGNAL_NAME,()=>{fs.appendFileSync(process.env.FIXTURE_EVENTS,JSON.stringify({kind:"child-stopped"})+"\\\\n");process.exit(0)});setTimeout(()=>process.exit(91),10000);console.log("ready");\'], {env: {...env, FIXTURE_SIGNAL_NAME: signal}, stdio: ["ignore", "pipe", "ignore"]});',
      '    process.on(signal, () => { event("worker-stopped"); child.once("exit", () => { setTimeout(() => { event("cache-exported"); event("runner-cleaned"); process.exit(0); }, Number(env.FIXTURE_CLEANUP_DELAY_MS || 0)); }); });',
      '    child.stdout.once("data", () => { event("child-started"); process.kill(Number(env.FIXTURE_DEPLOY_PID), callerSignal); });',
      "    setTimeout(() => process.exit(91), 10000);",
      "  } else {",
      "  if (build) {",
      '    if (env.FIXTURE_WINDOWS_FAIL === "1") process.exit(37);',
      '    if (env.FIXTURE_MISSING_ARCHIVE !== "1") fs.writeFileSync("mom-v9.8.7-win.zip", "fixture archive");',
      '  } else if (args.includes("gpu-multi")) console.log("ℹ tests 5\\nℹ pass 3\\nℹ fail 0\\nℹ skipped 2");',
      '  else console.log("MOM_TEST_SUMMARY 1 1 0 0");',
      "  process.exit(0);",
      "  }",
      "}",
      'if (tool === "docker") {',
      '  if (args[0] === "container") process.exit(1);',
      '  if (args[0] === "image") process.exit(0);',
      '  if (args[0] === "run") {',
      '    if (args.includes("--entrypoint")) { event("tools-probe", {safe: args.includes("none") && args.includes("--read-only") && args.includes("ALL") && args.includes("no-new-privileges:true"), complete: args.some(arg => arg.includes("command -v patchelf") && arg.includes("command -v readelf"))}); process.exit(env.FIXTURE_MISSING_TOOLS === "1" ? 1 : 0); }',
      '    if (!args.includes("mom-all")) { event("linux-lane"); console.log("MOM_TEST_SUMMARY 1 1 0 0"); process.exit(0); }',
      '    event("linux-build"); if (env.FIXTURE_SIGNAL && env.MOM_DEPLOY_TARGET === "linux") { if (env.FIXTURE_IGNORE_STOP === "1") process.on("SIGTERM", () => {}); process.kill(Number(env.FIXTURE_DEPLOY_PID), "SIG" + env.FIXTURE_SIGNAL); Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10000); }',
      // Two handshakes force both a sleep and a refresh inside Linux work, regardless of scheduling.
      '    if (env.FIXTURE_LONG_LINUX === "1") for (const requested of [1, 2]) { publish(env.FIXTURE_REFRESH_REQUEST, requested); const deadline = Date.now() + 10000; while (epoch(env.FIXTURE_CREDENTIAL) !== requested && !fs.existsSync(env.FIXTURE_REFRESH_FAILURE)) { if (Date.now() >= deadline) { event("fixture-timeout"); process.exit(91); } Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10); } if (fs.existsSync(env.FIXTURE_REFRESH_FAILURE)) break; event("credential-renewed"); }',
      '    process.exit(env.FIXTURE_LINUX_FAIL === "1" ? 41 : 0);',
      "  }",
      '  if (args[0] === "buildx" && args[1] === "build") { event("tools-upgrade", {incremental: args.includes("release-tools") && args.includes("MOM_RELEASE_TOOLS_BASE=mom-build-multicompiler") && args[args.indexOf("--builder") + 1] === "default"}); process.exit(env.FIXTURE_UPGRADE_FAIL === "1" ? 43 : 0); }',
      "  process.exit(0);",
      "}",
      'if (tool !== "fixture-win" && tool !== "sleep") process.exit(97);',
    ].join("\n");
    for (const tool of ["setsid", "git", "flock", "id", "chown", "nvidia-smi", "sleep", "cp", "sudo", "fixture-win", "docker"]) {
      if (tool === "setsid" && controls["FIXTURE_REAL_SETSID"] === "1") {
        fs.symlinkSync("/usr/bin/setsid", path.join(bin, tool));
      } else {
        fs.writeFileSync(path.join(bin, tool), stub, {mode: 0o755});
      }
    }
    if (controls["FIXTURE_EXTERNAL_SLEEP"] === "1") {
      // Preserve Bash's real signal/wait diagnostics: the Node sleep fixture
      // handles SIGTERM itself and cannot reproduce a terminated external sleep.
      fs.writeFileSync(path.join(bin, "sleep"), [
        "#!/usr/bin/env bash",
        'printf "[%s,%s]\\n" "$$" "$PPID" >>"$FIXTURE_HELPERS"',
        'printf \'{"kind":"refresh-wait"}\\n\' >>"$FIXTURE_EVENTS"',
        'exec /bin/sleep "$@"',
      ].join("\n"), {mode: 0o755});
    }
    try {
      if (controls["MOM_DEPLOY_REUSE_ARCHIVE"] === "1" && controls["FIXTURE_REUSE_MISSING"] !== "1") {
        fs.writeFileSync(path.join(temp, "mom-v9.8.7-win.zip"), "fixture archive");
      }
      const result = spawnSync("bash", ["-c", 'export FIXTURE_DEPLOY_PID=$$; exec bash "$1"',
        "fixture-deploy", path.join(temp, "scripts/test-deploy.sh")], {
        cwd: temp, encoding: "utf8", timeout: 15000,
        env: {PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`,
          HOME: temp, MOM_DEPLOY_TARGET: "windows", MOM_RELEASE_VERSION: "9.8.7",
          WIN_RUN: path.join(bin, "fixture-win"), WIN_MOM_DEV_BASE: "fixture-base",
          FIXTURE_ROOT: temp, FIXTURE_EVENTS: eventsPath, FIXTURE_HELPERS: helpersPath,
          FIXTURE_CREDENTIAL: credentialPath, FIXTURE_REFRESH_REQUEST: refreshRequestPath,
          FIXTURE_REFRESH_FAILURE: refreshFailurePath, ...controls},
      });
      assert.equal(result.error, undefined);
      /** @type {{kind: string, active?: boolean, captured?: boolean, noninteractive?: boolean, passwordInput?: boolean, exists?: boolean, safe?: boolean, complete?: boolean, incremental?: boolean, sameSession?: boolean, sameCallerGroup?: boolean, auth?: boolean, multi?: boolean, discreteGroup?: boolean}[]} */
      const events = fs.existsSync(eventsPath) ? fs.readFileSync(eventsPath, "utf8").trim().split("\n").map(line => JSON.parse(line)) : [];
      const summary = result.stdout.split("▶ Deployment summary\n")[1] ?? "";
      const lanes = summary.split("\n").filter(line => /^ {2}[✔✖➖] /.test(line));
      // Owned fixture helpers must be gone before removing their temporary files.
      // Read only the exact PIDs written by this fixture, never host processes.
      const helperPids = fs.existsSync(helpersPath) ? fs.readFileSync(helpersPath, "utf8")
        .trim().split("\n").flatMap(line => JSON.parse(line)) : [];
      for (const pid of new Set(helperPids)) {
        const statePath = `/proc/${pid}/stat`;
        assert.ok(!fs.existsSync(statePath) || /\) Z /.test(fs.readFileSync(statePath, "utf8")),
          "dispatcher exit must stop each owned credential helper");
      }
      return {status: result.status, events, lanes, output: result.stdout + result.stderr};
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  }

  /** @param {ReturnType<typeof deploy>} result @param {string[]} expected */
  function requestedOnce(result, expected) {
    const lanes = result.lanes.map(line => line.trim().split(" ")[1]);
    assert.deepEqual(lanes, expected);
    assert.equal(new Set(lanes).size, lanes.length);
  }

  const windows = ["intel-windows", "nvidia-windows", "amd-windows", "multi-windows"];
  const linux = ["intel-linux", "nvidia-linux", "amd-linux", "multi-linux"];
  test("Linux packaging failure records the mixed-vendor lane without running consumers", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "linux", FIXTURE_LINUX_FAIL: "1"});
    assert.equal(result.status, 1, result.output);
    requestedOnce(result, linux);
    assert.equal(result.events.filter(event => event.kind === "linux-lane").length, 0);
  });
  test("Windows mixed-vendor totals reflect actual suite tests and optional skips", () => {
    const result = deploy();
    assert.equal(result.status, 0, result.output);
    assert.match(result.output, /tests 8/);
    assert.match(result.output, /pass 6/);
    assert.match(result.output, /skipped 2/);
    const mixed = result.events.filter(event => event.multi);
    assert.equal(mixed.length, 1);
    assert.equal(mixed[0]?.discreteGroup, true);
  });
  test("interrupted Linux packaging records every pending lane including mixed-vendor", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "linux", FIXTURE_SIGNAL: "TERM"});
    assert.equal(result.status, 143, result.output);
    requestedOnce(result, linux);
    assert.ok(result.lanes.every(line => line.includes("interrupted")));
    assert.equal(result.events.filter(event => event.kind === "linux-lane").length, 0);
  });
  test("clean deployment rejects archive reuse and malformed rebuild controls before dispatch", () => {
    for (const controls of [
      {MOM_FORCE_REBUILD: "1", MOM_DEPLOY_REUSE_ARCHIVE: "1"},
      {MOM_FORCE_REBUILD: "yes"},
    ]) {
      const result = deploy(controls);
      assert.equal(result.status, 2);
      assert.match(result.output, /MOM_FORCE_REBUILD/);
      assert.equal(result.events.length, 0);
    }
  });
  for (const target of ["windows", "all", "windows-nvidia"]) {
    test(`${target}: failed Windows packaging never dispatches archive consumers`, () => {
      const result = deploy({MOM_DEPLOY_TARGET: target, FIXTURE_WINDOWS_FAIL: "1", FIXTURE_LINUX_FAIL: "1"});
      assert.notEqual(result.status, 0);
      assert.equal(result.events.filter(event => event.kind === "windows-build").length, 1);
      assert.equal(result.events.filter(event => event.kind === "windows-lane" || event.kind === "copy" || event.kind === "chown").length, 0);
      assert.equal(result.events.filter(event => event.kind === "sudo").length, 1);
      requestedOnce(result, target === "all" ? [...linux, ...windows]
        : target === "windows" ? windows : ["nvidia-windows"]);
    });
  }

  test("successful Windows build with no downloaded ZIP fails before copy or lane sudo", () => {
    const result = deploy({FIXTURE_MISSING_ARCHIVE: "1"});
    assert.notEqual(result.status, 0);
    assert.equal(result.events.filter(event => event.kind === "windows-lane" || event.kind === "copy" || event.kind === "chown").length, 0);
    assert.equal(result.events.filter(event => event.kind === "sudo").length, 1);
    requestedOnce(result, windows);
  });

  test("Windows staging copy failure never launches any archive consumer", () => {
    const result = deploy({FIXTURE_COPY_FAIL: "1"});
    assert.notEqual(result.status, 0);
    assert.equal(result.events.filter(event => event.kind === "windows-build").length, 1);
    assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 0);
    assert.equal(result.events.filter(event => event.kind === "copy").length, 4);
    assert.ok(result.lanes.every(line => line.includes("✖")));
    requestedOnce(result, windows);
  });

  test("Linux build failure does not suppress independently configured Windows lanes", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_LINUX_FAIL: "1"});
    assert.notEqual(result.status, 0);
    assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 4);
    requestedOnce(result, [...linux, ...windows]);
  });

  test.describe("Windows credential lifetime and owned cleanup", () => {
    test("failed Windows packaging quietly stops a real external refresh sleep", () => {
      const result = deploy({FIXTURE_EXTERNAL_SLEEP: "1", FIXTURE_WINDOWS_FAIL: "1"});
      assert.equal(result.status, 1, result.output);
      assert.equal(result.events.filter(event => event.kind === "refresh-wait").length, 1);
      assert.equal(result.events.filter(event => event.kind === "windows-build").length, 1);
      assert.match(result.output, /Build Windows release \(exit 37\)/);
      assert.doesNotMatch(result.output, /Terminated|sleep 60|Killed/,
        "expected helper teardown must not obscure the actual deployment failure");
      requestedOnce(result, windows);
    });

    test("Windows authentication validates outside the captured process group", () => {
      const result = deploy();
      assert.equal(result.status, 0);
      const auth = result.events.filter(event => event.kind === "auth");
      assert.ok(auth.length >= 1, "validate the timestamp before dispatch");
      assert.ok(auth.every(event => !event.active && !event.captured && event.noninteractive));
      requestedOnce(result, windows);
    });

    test("combined deployment authenticates before starting expensive Linux work", () => {
      const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_LINUX_FAIL: "1"});
      const authAt = result.events.findIndex(event => event.kind === "auth");
      const linuxAt = result.events.findIndex(event => ["tools-probe", "linux-build"].includes(event.kind));
      assert.ok(authAt >= 0 && authAt < linuxAt, "credential admission must precede Linux build admission");
      assert.ok(result.events.filter(event => event.kind === "sudo").every(event => event.noninteractive),
        "Windows phases must never start a late password prompt");
    });

    test("noninteractive refresh retains credentials through simulated long Linux work", {
      skip: !fs.existsSync("/usr/bin/setsid") || !fs.existsSync("/usr/bin/ps"),
    }, () => {
      const groups = spawnSync("/usr/bin/ps", ["-o", "sid=,pgid=", "-p", String(process.pid)], {encoding: "utf8"});
      assert.equal(groups.status, 0);
      const [sid, pgid] = groups.stdout.trim().split(/\s+/);
      assert.ok(sid && pgid);
      const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_LONG_LINUX: "1",
        FIXTURE_REAL_SESSION: "1", FIXTURE_EXPECTED_SID: sid, FIXTURE_EXPECTED_PGID: pgid});
      assert.equal(result.status, 0, result.output + "\n" + JSON.stringify(result.events));
      const linuxAt = result.events.findIndex(event => event.kind === "linux-build");
      const windowsAt = result.events.findIndex(event => event.kind === "windows-build");
      const duringLinux = result.events.slice(linuxAt + 1, windowsAt);
      assert.ok(duringLinux.some(event => event.kind === "refresh-wait"));
      assert.ok(duringLinux.some(event => event.kind === "auth" && event.noninteractive));
      assert.equal(duringLinux.filter(event => event.kind === "credential-renewed").length, 2);
      assert.equal(result.events.filter(event => event.kind === "credential-expired").length, 0);
      assert.ok(result.events.filter(event => event.kind === "auth" || event.kind === "sudo")
        .every(event => event.noninteractive), "refresh and Windows dispatch must never prompt");
      const authSessions = result.events.filter(event => event.kind === "session" && event.auth);
      assert.ok(authSessions.every(event => event.sameSession), "refresh must retain the caller's sudo timestamp session");
      assert.ok(authSessions.some(event => !event.sameCallerGroup), "the refresher must own a separate cleanup group");
      requestedOnce(result, [...linux, ...windows]);
    });

    test("lost refresh credentials fail Windows admission without a late interactive prompt", () => {
      const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_LONG_LINUX: "1", FIXTURE_REFRESH_FAIL: "1"});
      assert.notEqual(result.status, 0);
      assert.ok(result.events.some(event => event.kind === "refresh-failed"));
      assert.equal(result.events.filter(event => event.kind.startsWith("windows-") || event.kind === "sudo").length, 0);
      assert.ok(result.events.filter(event => event.kind === "auth").every(event => event.noninteractive));
      requestedOnce(result, [...linux, ...windows]);
    });

    test("Windows workers retain the caller session while owning isolated process groups", {
      skip: !fs.existsSync("/usr/bin/setsid") || !fs.existsSync("/usr/bin/ps"),
    }, () => {
      const groups = spawnSync("/usr/bin/ps", ["-o", "sid=,pgid=", "-p", String(process.pid)], {encoding: "utf8"});
      assert.equal(groups.status, 0);
      const [sid, pgid] = groups.stdout.trim().split(/\s+/);
      assert.ok(sid && pgid);
      const result = deploy({FIXTURE_REAL_SESSION: "1", FIXTURE_EXPECTED_SID: sid, FIXTURE_EXPECTED_PGID: pgid});
      assert.equal(result.status, 0);
      const sessions = result.events.filter(event => event.kind === "session");
      const workers = sessions.filter(event => !event.auth);
      assert.equal(workers.length, 5);
      assert.ok(sessions.every(event => event.sameSession), "sudo timestamps must retain the caller's terminal session");
      assert.ok(workers.every(event => !event.sameCallerGroup), "interrupts must target only the owned worker process group");
      requestedOnce(result, windows);
    });

    for (const [signal, status] of [["TERM", 143], ["INT", 130]]) {
      test(`${signal} stops the owned Windows worker and descendant before finishing once`, () => {
        const result = deploy({FIXTURE_SIGNAL: String(signal)});
        assert.equal(result.status, status, result.output);
        for (const kind of ["child-started", "child-stopped", "worker-stopped"]) {
          assert.equal(result.events.filter(event => event.kind === kind).length, 1, `${kind} must complete once`);
        }
        assert.equal(result.events.filter(event => event.kind === "windows-lane" || event.kind === "copy" || event.kind === "chown").length, 0);
        assert.ok(result.lanes.every(line => line.includes("interrupted")));
        requestedOnce(result, windows);
      });
      test(`${signal} waits for Windows cache export and teardown beyond Linux kill grace`, () => {
        const result = deploy({FIXTURE_SIGNAL: String(signal), FIXTURE_SHORT_INTERRUPT_GRACE: "1",
          FIXTURE_CLEANUP_DELAY_MS: "2000"});
        assert.equal(result.status, status, result.output);
        const exportAt = result.events.findIndex(event => event.kind === "cache-exported");
        const cleanupAt = result.events.findIndex(event => event.kind === "runner-cleaned");
        assert.ok(exportAt > result.events.findIndex(event => event.kind === "child-stopped"));
        assert.ok(cleanupAt > exportAt);
        assert.equal(result.events.filter(event => event.kind === "runner-cleaned").length, 1);
        assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 0);
        assert.ok(result.lanes.every(line => line.includes("interrupted")));
        requestedOnce(result, windows);
      });
    }

    test("TERM retains bounded Linux escalation when the owned worker ignores termination", {
      skip: !fs.existsSync("/usr/bin/setsid"),
    }, () => {
      const started = Date.now();
      const result = deploy({MOM_DEPLOY_TARGET: "linux", FIXTURE_SIGNAL: "TERM",
        FIXTURE_SHORT_INTERRUPT_GRACE: "1", FIXTURE_IGNORE_STOP: "1", FIXTURE_REAL_SETSID: "1"});
      assert.ok(Date.now() - started < 8000, "Linux escalation must precede the worker's 10-second fallback exit");
      assert.equal(result.status, 143, result.output);
      assert.equal(result.events.filter(event => event.kind === "linux-build").length, 1);
      assert.equal(result.events.filter(event => event.kind === "linux-lane" || event.kind === "windows-build").length, 0);
      requestedOnce(result, linux);
    });

    test("noninteractive authentication failure is actionable and never starts Linux or the VM", () => {
      const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_AUTH_FAIL: "1"});
      assert.notEqual(result.status, 0);
      assert.equal(result.events.filter(event => event.kind.startsWith("windows-") || event.kind === "copy" || event.kind === "sudo").length, 0);
      assert.equal(result.events.filter(event => ["tools-probe", "linux-build", "linux-lane", "refresh-wait"].includes(event.kind)).length, 0);
      assert.match(result.output, /sudo -v|SUDO_PASSWORD/);
      assert.equal(result.events.find(event => event.kind === "auth")?.active, false);
      requestedOnce(result, [...linux, ...windows]);
    });

    test("explicit fixture password is accepted without entering captured diagnostics", () => {
      const result = deploy({SUDO_PASSWORD: "fixture-only-auth-input"});
      assert.equal(result.status, 0);
      assert.ok(result.events.filter(event => event.kind === "sudo").every(event => event.passwordInput));
      assert.equal(result.events.filter(event => event.kind === "auth").length, 0);
      assert.equal(result.events.filter(event => event.kind === "refresh-wait").length, 0);
      assert.doesNotMatch(result.output, /fixture-only-auth-input/);
      requestedOnce(result, windows);
    });

    test("reused Windows archive still authenticates before lane dispatch", () => {
      const result = deploy({MOM_DEPLOY_REUSE_ARCHIVE: "1"});
      assert.equal(result.status, 0);
      assert.equal(result.events.filter(event => event.kind === "windows-build").length, 0);
      assert.ok(result.events.filter(event => event.kind === "auth").length >= 1);
      assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 4);
      requestedOnce(result, windows);
    });

    test("single vendor deployment never schedules mixed Windows work", () => {
      const result = deploy({MOM_DEPLOY_TARGET: "windows-intel"});
      assert.equal(result.status, 0);
      assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 1);
      requestedOnce(result, ["intel-windows"]);
    });

    test("unavailable optional Windows runner skips every requested lane without authentication", () => {
      const result = deploy({WIN_RUN: "/fixture/no-windows-runner"});
      assert.equal(result.status, 0);
      assert.equal(result.events.filter(event => event.kind === "auth" || event.kind === "sudo" || event.kind === "copy").length, 0);
      assert.equal(result.events.filter(event => event.kind === "refresh-wait").length, 0);
      assert.ok(result.lanes.every(line => line.includes("➖")));
      requestedOnce(result, windows);
    });

    test("already-root deployment does not require sudo or an authentication prompt", () => {
      const result = deploy({FIXTURE_ROOT_USER: "1"});
      assert.equal(result.status, 0);
      assert.equal(result.events.filter(event => event.kind === "auth" || event.kind === "sudo").length, 0);
      assert.equal(result.events.filter(event => event.kind === "refresh-wait").length, 0);
      assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 4);
      requestedOnce(result, windows);
    });
  });

  test("healthy cached Linux image is probed without rebuilding compiler stages", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "linux-nvidia"});
    assert.equal(result.status, 0, result.output);
    assert.equal(result.events.filter(event => event.kind === "tools-probe").length, 1);
    assert.equal(result.events.find(event => event.kind === "tools-probe")?.safe, true);
    assert.equal(result.events.find(event => event.kind === "tools-probe")?.complete, true);
    assert.equal(result.events.filter(event => event.kind === "tools-upgrade").length, 0);
    assert.equal(result.events.filter(event => event.kind === "linux-build").length, 1);
    assert.equal(result.events.filter(event => ["auth", "sudo", "refresh-wait"].includes(event.kind)).length, 0);
    requestedOnce(result, ["nvidia-linux"]);
  });

  test("cached Linux image lacking release tools receives one repair before build", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "linux-nvidia", FIXTURE_MISSING_TOOLS: "1"});
    assert.equal(result.status, 0, result.output);
    const lifecycle = result.events.filter(event => ["tools-probe", "tools-upgrade", "linux-build"].includes(event.kind));
    assert.deepEqual(lifecycle.map(event => event.kind), ["tools-probe", "tools-upgrade", "linux-build"]);
    assert.equal(lifecycle[1]?.incremental, true, "repair only the release-tools layer atop the cached image");
    requestedOnce(result, ["nvidia-linux"]);
  });

  test("failed cached Linux tools repair stops build but retains independent Windows deployment", () => {
    const result = deploy({MOM_DEPLOY_TARGET: "all", FIXTURE_MISSING_TOOLS: "1", FIXTURE_UPGRADE_FAIL: "1"});
    assert.notEqual(result.status, 0);
    assert.equal(result.events.filter(event => event.kind === "tools-upgrade").length, 1);
    assert.equal(result.events.filter(event => event.kind === "linux-build").length, 0);
    assert.equal(result.events.filter(event => event.kind === "windows-lane").length, 4);
    requestedOnce(result, [...linux, ...windows]);
  });
});

test.describe("Benchmark and installer scripts", {concurrency: false}, () => {
  const releaseFixture = path.join(__dirname, "fixtures", "release_benchmark_worker.js");
  // Individual notice tests replace this default sink with their own collectors.
  let previousFixtureError = console.error;
  test.beforeEach(() => {
    previousFixtureError = console.error;
    console.error = () => undefined;
  });
  test.afterEach(() => {console.error = previousFixtureError;});

  /** @param {string} mode @param {string} [algo] @param {string} [platform]
 * @param {number} [timeoutMs] @param {string} [evidenceRoot] @param {string} [additionalAlgo]
 * @param {string} [minerFixture] @param {boolean} [debug] */
  async function runReleaseFixture(mode, algo = "fake", platform = "intel-linux", timeoutMs = 30000,
    evidenceRoot, additionalAlgo, minerFixture = releaseFixture, debug = false) {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-benchmark-test-"));
    const readme = path.join(temp, "README.md");
    const gpu = platform.startsWith("intel-") ? "B580" :
      platform.startsWith("nvidia-") ? "5060 Ti" : "9060 XT";
    fs.writeFileSync(readme, [
      "| Algo / coin / pool | GPU | Linux | Windows |",
      "| --- | --- | --- | --- |",
      `| \`${algo}\` | ${gpu} | 100 H/s | 100 H/s |`,
      ...(additionalAlgo ? [`| \`${additionalAlgo}\` | ${gpu} | 100 H/s | 100 H/s |`] : []),
      "",
    ].join("\n"));
    const previous = process.env["MOM_RELEASE_FIXTURE_MODE"];
    const previousAlgo = process.env["MOM_RELEASE_FIXTURE_ALGO"];
    const previousEvidence = process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
    const previousDebug = process.env["MOM_DEPLOY_DEBUG"];
    const previousLog = console.log;
    /** @type {string[]} */
    const output = [];
    console.log = (...args) => output.push(args.join(" "));
    process.env["MOM_RELEASE_FIXTURE_MODE"] = mode;
    process.env["MOM_RELEASE_FIXTURE_ALGO"] = algo;
    if (debug) {
      process.env["MOM_DEPLOY_DEBUG"] = "1";
    } else {
      delete process.env["MOM_DEPLOY_DEBUG"];
    }
    if (evidenceRoot !== undefined) {
      process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"] = evidenceRoot;
    }
    try {
      const result = await release.main(release.parseOptions([
        "--platform", platform, "--miner", minerFixture, "--readme", readme,
        "--timeout-ms", String(timeoutMs),
      ]));
      const summaries = output.filter(line => /^ {2}[✔✖] /u.test(line));
      assert.equal(summaries.length, result.tests);
      assert.ok(summaries.every(line => /^ {2}[✔✖] [a-z0-9/_-]+: /u.test(line)),
        "fixture must emit one recognizable summary per algorithm");
      assert.equal(output.filter(line => /^\[perf\] .*: starting benchmark \(deadline \d+ ms\)$/.test(line)).length,
        debug ? result.tests : 0, debug ? "every benchmark must announce its start" :
          "benchmark start chatter is opt-in");
      return result;
    } finally {
      console.log = previousLog;
      if (previous === undefined) {
        delete process.env["MOM_RELEASE_FIXTURE_MODE"];
      } else {
        process.env["MOM_RELEASE_FIXTURE_MODE"] = previous;
      }
      if (previousAlgo === undefined) {
        delete process.env["MOM_RELEASE_FIXTURE_ALGO"];
      } else {
        process.env["MOM_RELEASE_FIXTURE_ALGO"] = previousAlgo;
      }
      if (previousEvidence === undefined) {
        delete process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
      } else {
        process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"] = previousEvidence;
      }
      if (previousDebug === undefined) {
        delete process.env["MOM_DEPLOY_DEBUG"];
      } else {
        process.env["MOM_DEPLOY_DEBUG"] = previousDebug;
      }
      fs.rmSync(temp, {recursive: true, force: true});
    }
  }

  /** @param {string} file */
  function sha256(file) {
    return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  }

  test.describe("Benchmark options and process lifecycle", () => {
    test("GPU benchmark options are explicit and bounded", () => {
      const options = benchmark.parseOptions([
        "--output", "result.json", "--samples", "3", "--warmup-samples", "2",
        "--timeout-ms", "30000", "--algos", "zhash, nexapow", "--backend", "sycl",
        "--dev", "gpu1*[intensity=1]",
      ], "linux", "x64");
      assert.equal(options.outputPath, path.resolve("result.json"));
      assert.equal(options.label, "linux-x64");
      assert.equal(options.samplesWanted, 3);
      assert.equal(options.warmupSamples, 2);
      assert.equal(options.timeoutMs, 30000);
      assert.deepEqual(options.requested, ["zhash", "nexapow"]);
      assert.equal(options.backend, "sycl");
      assert.equal(options.dev, "gpu1*[intensity=1]");
      assert.throws(() => benchmark.parseOptions(["--unknown", "x"]), /Unknown option/);
      assert.throws(() => benchmark.parseOptions(["--samples", "0"]), /at least 1/);
      assert.throws(() => benchmark.parseOptions(["--timeout-ms", "NaN"]), /at least 30000/);
      assert.throws(() => benchmark.parseOptions(["--backend", "invalid"]), /Invalid GPU backend/);
    });

    test("GPU benchmark defaults discard cold output and retain three steady samples", () => {
      const options = benchmark.parseOptions([]);
      assert.equal(options.samplesWanted, 3);
      assert.equal(options.warmupSamples, 1);
    });

    test("GPU benchmark graceful-only mode disables signal fallbacks", () => {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/benchmark-gpu-algos.js"), "utf8");
      assert.match(source,
        /const gracefulOnly = process\.env\["MOM_BENCHMARK_GRACEFUL_ONLY"\] === "1";/);
      assert.match(source,
        /else if \(!gracefulOnly\) \{\s*killProcessTree\(child, "SIGTERM"\);\s*\}/);
      assert.match(source,
        /if \(!gracefulOnly\) \{\s*forceTimer = setTimeout\(\(\) => killProcessTree\(child, "SIGKILL"\), 10000\);\s*\}/);
    });

    test("GPU benchmark graceful-only discovery waits for natural runtime initialization", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-discovery-timeout-test-"));
      const previousDirectory = process.cwd();
      const previousGraceful = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const options = {
        outputPath: path.join(temp, "report.json"), label: "discovery-fixture",
        samplesWanted: 1, warmupSamples: 0, timeoutMs: 50, requested: [],
        backend: undefined, dev: undefined,
      };
      fs.writeFileSync(path.join(temp, "mom.js"), `
    "use strict";
    if (process.argv[2] === "algorithms") {
      setTimeout(() => process.stdout.write("MOM_ALGORITHMS { }\\n"), 300);
    } else {
      process.exitCode = 2;
    }
  `);
      const run = () => benchmark.main(options);
      try {
        process.chdir(temp);
        process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
        await assert.rejects(run(), /algorithms did not report any GPU jobs/);

        delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        await assert.rejects(run(), /algorithms failed:.*(?:ETIMEDOUT|timed out|SIGTERM)/i);

        process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "0";
        await assert.rejects(run(), /algorithms failed:.*(?:ETIMEDOUT|timed out|SIGTERM)/i);
      } finally {
        process.chdir(previousDirectory);
        if (previousGraceful === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previousGraceful;
        }
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("GPU benchmark runner discovers jobs, samples rates, and closes the miner cleanly", () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-benchmark-test-"));
      const outputPath = path.join(temp, "missing", "report.json");
      fs.writeFileSync(path.join(temp, "mom.js"), `
    "use strict";
    if (process.argv[2] === "algorithms") {
      process.stdout.write('MOM_ALGORITHMS {"fake":"gpu1"}\\n');
    } else if (process.argv[2] === "bench") {
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (data) => {
        if (data.includes("close")) { process.exit(0); }
      });
      for (let sample = 0; sample < 3; sample += 1) {
        console.log("Algo fake (gpu1:auto[sycl]) hashrate: 1.00 KH/s");
      }
      process.stderr.write("LOOPSTAT t=123 wall=10.0s cpu=4.5% dispatch=99.0% " +
        "msg=0.001s post=0.002s iters=3 jobs=1 max_msg=0.2ms max_post=0.3ms\\n");
    } else {
      process.exitCode = 2;
    }
  `);
      try {
        // Capture in a child: replacing this test process's stdout also intercepts Node's reporter.
        const output = spawnSync(process.execPath, [require.resolve("../scripts/benchmark-gpu-algos"),
          "--output", outputPath, "--label", "fixture", "--samples", "2", "--warmup-samples", "1",
          "--timeout-ms", "30000", "--algos", "fake", "--dev", "gpu1*[workgroup=256]",
        ], {cwd: temp, encoding: "utf8", timeout: 5000, env: {...process.env, MOM_LOOP_STATS: "1"}});
        assert.equal(output.error, undefined);
        assert.equal(output.signal, null);
        assert.equal(output.status, 0, output.stderr);
        assert.equal(output.stdout, "\n=== fixture: fake (gpu1*[workgroup=256]) ===\n" +
        "Algo fake (gpu1:auto[sycl]) hashrate: 1.00 KH/s\n".repeat(3) +
        `\nWrote ${outputPath}\n`);
        assert.equal(output.stderr, "LOOPSTAT t=123 wall=10.0s cpu=4.5% dispatch=99.0% " +
        "msg=0.001s post=0.002s iters=3 jobs=1 max_msg=0.2ms max_post=0.3ms\n");
        const report = JSON.parse(fs.readFileSync(outputPath, "utf8"));
        assert.deepEqual(report.jobs, {fake: "gpu1*[workgroup=256]"});
        assert.equal(report.results[0]?.status, "ok");
        assert.equal(report.results[0]?.warmup_samples.length, 1);
        assert.equal(report.results[0]?.samples.length, 2);
        assert.deepEqual(report.results[0]?.loop_stats, [{
          timestamp: 123, wall_seconds: 10, cpu_percent: 4.5, dispatch_percent: 99,
          message_seconds: 0.001, post_seconds: 0.002, iterations: 3, jobs: 1,
          max_message_ms: 0.2, max_post_ms: 0.3,
        }]);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("GPU benchmark runner stops promptly on a split compute-core error", () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-benchmark-error-test-"));
      const outputPath = path.join(temp, "report.json");
      fs.writeFileSync(path.join(temp, "mom.js"), `
    "use strict";
    if (process.argv[2] === "algorithms") {
      process.stdout.write('MOM_ALGORITHMS {"fake":"gpu1"}\\n');
    } else if (process.argv[2] === "bench") {
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (data) => {
        if (data.includes("close")) { process.exit(0); }
      });
      process.stdout.write("ERROR: Compute core ");
      setTimeout(() => process.stdout.write("error: fixture\\n"), 10);
    } else {
      process.exitCode = 2;
    }
  `);
      try {
        const started = Date.now();
        const output = spawnSync(process.execPath, [require.resolve("../scripts/benchmark-gpu-algos"),
          "--output", outputPath, "--label", "fixture", "--samples", "1", "--warmup-samples", "0",
          "--timeout-ms", "30000", "--algos", "fake",
        ], {cwd: temp, encoding: "utf8", timeout: 5000});
        assert.equal(output.error, undefined);
        assert.equal(output.signal, null);
        assert.equal(output.status, 1);
        assert.equal(output.stdout, "\n=== fixture: fake (gpu1) ===\n" +
        "ERROR: Compute core error: fixture\n");
        assert.match(output.stderr, /fixture failed: fake=compute-error/);
        assert.ok(Date.now() - started < 5000);
        const report = JSON.parse(fs.readFileSync(outputPath, "utf8"));
        assert.equal(report.results[0]?.status, "compute-error");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release gate options reject ineffective limits", () => {
      const options = release.parseOptions([
        "--platform", "intel-linux", "--margin", "0.1", "--timeout-ms", "1",
      ]);
      assert.equal(options.platform, "intel-linux");
      assert.equal(options.margin, 0.1);
      assert.equal(options.timeoutMs, 1);
      assert.throws(() => release.parseOptions([]), /--platform/);
      assert.throws(() => release.parseOptions([
        "--platform", "intel-linux", "--timeout-ms", "NaN",
      ]), /positive integer/);
      assert.throws(() => release.parseOptions([
        "--platform", "intel-linux", "--margin", "1",
      ]), /between zero and one/);
      assert.throws(() => release.parseOptions([
        "--platform", "intel-linux", "--dev", "gpu1 & whoami",
      ]), /--dev invalid device entry/);
      assert.throws(() => release.parseOptions(["--platform", "intel-linux", "extra"]), /positional/);
    });

    test("release gate defaults to a five-percent regression margin", () => {
      assert.equal(release.parseOptions(["--platform", "intel-linux"]).margin, 0.05);
    });

    test("release timeout emits an immediate sanitized cleanup notice", async () => {
      const previousGraceful = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-timeout-evidence-test-"));
      /** @type {string[]} */
      const notices = [];
      const previousError = console.error;
      process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "0";
      console.error = (...args) => notices.push(args.join(" "));
      try {
        await assert.rejects(runReleaseFixture("timeout", "fake", "intel-linux", 2000, temp),
          /Timed out after 2000 ms:[\s\S]*fixture setup started[\s\S]*fixture runtime still initializing/);
        const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.verdict.passed, false);
        assert.equal(metadata.verdict.processOk, false);
        assert.match(metadata.verdict.failure, /^Timed out after 2000 ms:/);
      } finally {
        console.error = previousError;
        if (previousGraceful === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previousGraceful;
        }
        fs.rmSync(temp, {recursive: true, force: true});
      }
      assert.deepEqual(notices, [
        "Release command deadline reached after 2000 ms; stopping the command.",
      ]);
    });

    test("release graceful-only close waits for natural worker teardown without a kill deadline", async (t) => {
      const previous = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const previousError = console.error;
      const previousSetTimeout = globalThis.setTimeout;
      /** @type {string[]} */
      const notices = [];
      /** @param {() => void} callback @param {number} [delay] @returns {NodeJS.Timeout} */
      function scaledSetTimeout(callback, delay) {
        if (delay === 10 * 60 * 1000) {return previousSetTimeout(callback, 30);}
        return previousSetTimeout(callback, delay === 10000 ? 10 : delay);
      }
      process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
      console.error = (...args) => notices.push(args.join(" "));
      // Keep the child lifecycle real while making the existing ten-second parent warning timer
      // observable in this focused fake-child test.
      t.mock.method(globalThis, "setTimeout", scaledSetTimeout);
      try {
        const summary = await runReleaseFixture("slow-close", "fake", "intel-linux", 2000);
        assert.deepEqual(summary, {tests: 1, passed: 1, skipped: 0});
      } finally {
        console.error = previousError;
        if (previous === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previous;
        }
      }
      assert.deepEqual(notices, [
        "Release benchmark still waiting for cooperative cleanup.",
      ]);
    });

    test("release lifecycle chatter is opt-in deployment debug", async () => {
      const previousError = console.error;
      /** @type {string[]} */
      const notices = [];
      console.error = (...args) => notices.push(args.join(" "));
      try {
        await runReleaseFixture("slow-close");
        assert.equal(notices.some((notice) => notice.startsWith("[perf] ")), false);

        notices.length = 0;
        await runReleaseFixture("slow-close", "fake", "intel-linux", 2000,
          undefined, undefined, releaseFixture, true);
        assert.ok(notices.includes("[perf] fake: requesting cooperative close"));
      } finally {
        console.error = previousError;
      }
    });

    for (const silent of [false, true]) {
      test(`release graceful-only deadline closes ${silent ? "silent" : "initializing"} worker with live evidence before exit`, async () => {
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-live-deadline-test-"));
        const worker = path.join(temp, "worker.js");
        const evidence = path.join(temp, "evidence");
        const previousGraceful = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        const previousError = console.error;
        /** @type {string[]} */
        const notices = [];
        fs.writeFileSync(worker, `
          const fs=require('node:fs'),path=require('node:path');
          if(process.argv[2]==='algorithms') {
            console.log('gpu0:fixture\\nMOM_ALGORITHMS {"fake":"gpu0"}');process.exit(0);
          }
          for(const signal of ['SIGINT','SIGTERM']) process.on(signal,()=>process.exit(97));
          ${silent ? "" : "process.stdout.write('fixture setup started\\n');process.stderr.write('fixture initializing\\n');"}
          let input='';process.stdin.setEncoding('utf8');
          process.stdin.on('data',chunk=>{
            input+=chunk;if(input!=='close\\n') return;
            const root=process.env.MOM_RELEASE_PERF_EVIDENCE_DIR;
            const stdout=fs.readFileSync(path.join(root,'fake.stdout.log'),'utf8');
            const stderr=fs.readFileSync(path.join(root,'fake.stderr.log'),'utf8');
            if(${silent ? "stdout!==''||stderr!==''" : "!stdout.includes('fixture setup started')||!stderr.includes('fixture initializing')"}) process.exit(96);
            if(fs.existsSync(path.join(root,'fake.metadata.json'))) process.exit(95);
            console.log('fixture live evidence verified before close');process.exit(0);
          });
          setTimeout(()=>process.exit(91),2000);
        `);
        process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
        console.error = (...args) => notices.push(args.join(" "));
        try {
          await assert.rejects(runReleaseFixture("clean", "fake", "intel-linux", 500, evidence, undefined, worker),
            /Timed out after 500 ms:[\s\S]*fixture live evidence verified before close/);
          const metadata = JSON.parse(fs.readFileSync(path.join(evidence, "fake.metadata.json"), "utf8"));
          assert.equal(metadata.exitCode, 0);
          assert.equal(metadata.signal, null);
          assert.equal(metadata.gracefulClose, true);
          assert.equal(metadata.verdict.passed, false);
          assert.match(metadata.verdict.failure, /^Timed out after 500 ms:/);
          assert.match(notices.join("\n"), /requesting cooperative close/);
        } finally {
          console.error = previousError;
          if (previousGraceful === undefined) {delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];} else {
            process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previousGraceful;
          }
          fs.rmSync(temp, {recursive: true, force: true});
        }
      });
    }

    test("release graceful-only deadline requests control without native signals", async (t) => {
      const previous = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const previousError = console.error;
      const previousSetTimeout = globalThis.setTimeout;
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-graceful-deadline-test-"));
      let commandDeadlineCount = 0;
      /** @type {string[]} */
      const notices = [];
      /** @param {() => void} callback @param {number} [delay] @returns {NodeJS.Timeout} */
      function scaledSetTimeout(callback, delay) {
        if (delay === 2000) {
          commandDeadlineCount++;
          // Discovery uses the same deadline as the benchmark. Let that short-lived process finish;
          // only the benchmark deadline belongs to this timeout-path test.
          return previousSetTimeout(callback, commandDeadlineCount === 1 ? delay : 5);
        }
        if (delay === 10000) {return previousSetTimeout(callback, 10);}
        if (delay === 10 * 60 * 1000) {return previousSetTimeout(callback, 30);}
        return previousSetTimeout(callback, delay);
      }
      process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
      console.error = (...args) => notices.push(args.join(" "));
      t.mock.method(globalThis, "setTimeout", scaledSetTimeout);
      try {
        await assert.rejects(runReleaseFixture("timeout-finite", "fake", "intel-linux", 2000, temp),
          /Timed out after 2000 ms:[\s\S]*fixture finite work complete[\s\S]*fixture control input received/);
        const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.exitCode, 0);
        assert.equal(metadata.signal, null);
        assert.equal(metadata.gracefulClose, true);
        assert.equal(metadata.verdict.passed, false);
      } finally {
        console.error = previousError;
        if (previous === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previous;
        }
        fs.rmSync(temp, {recursive: true, force: true});
      }
      assert.deepEqual(notices, [
        "Release command deadline reached after 2000 ms; requesting cooperative close; waiting for native cleanup.",
        "Release benchmark still waiting for cooperative cleanup.",
      ]);
    });

    test("release graceful-only deadline closes before late samples without promoting the window", async (t) => {
      const previous = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const previousError = console.error;
      const previousSetTimeout = globalThis.setTimeout;
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-late-samples-test-"));
      let commandDeadlineCount = 0;
      /** @type {string[]} */
      const notices = [];
      /** @param {() => void} callback @param {number} [delay] @returns {NodeJS.Timeout} */
      function scaledSetTimeout(callback, delay) {
        if (delay === 2000) {
          commandDeadlineCount++;
          return previousSetTimeout(callback, commandDeadlineCount === 1 ? delay : 5);
        }
        return previousSetTimeout(callback, delay);
      }
      process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
      console.error = (...args) => notices.push(args.join(" "));
      t.mock.method(globalThis, "setTimeout", scaledSetTimeout);
      try {
        await assert.rejects(runReleaseFixture("late-samples", "fake", "intel-linux", 2000, temp),
          /Timed out after 2000 ms:/);
        const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.exitCode, 0);
        assert.equal(metadata.signal, null);
        assert.equal(metadata.gracefulClose, true);
        assert.equal(metadata.sampleCount, 0);
        assert.equal(metadata.verdict.passed, false);
      } finally {
        console.error = previousError;
        if (previous === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previous;
        }
        fs.rmSync(temp, {recursive: true, force: true});
      }
      assert.deepEqual(notices, [
        "Release command deadline reached after 2000 ms; requesting cooperative close; waiting for native cleanup.",
      ]);
    });

    test("release graceful-only deadline allows one explicit interrupt close and retains timeout failure", async (t) => {
      const previous = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
      const previousError = console.error;
      const previousSetTimeout = globalThis.setTimeout;
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-deadline-interrupt-test-"));
      let commandDeadlineCount = 0;
      /** @type {string[]} */
      const notices = [];
      /** @param {() => void} callback @param {number} [delay] @returns {NodeJS.Timeout} */
      function scaledSetTimeout(callback, delay) {
        if (delay === 2000) {
          commandDeadlineCount++;
          if (commandDeadlineCount === 2) {
            return previousSetTimeout(() => {
              callback();
              // Exercise the registered interrupt handlers; send no OS or child signal.
              process.emit("SIGINT");
              process.emit("SIGTERM");
            }, 5);
          }
        }
        return previousSetTimeout(callback, delay);
      }
      process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
      console.error = (...args) => notices.push(args.join(" "));
      t.mock.method(globalThis, "setTimeout", scaledSetTimeout);
      try {
        await assert.rejects(runReleaseFixture("timeout-interrupt", "fake", "intel-linux", 2000, temp),
          /Timed out after 2000 ms:[\s\S]*fixture control close accepted[\s\S]*fixture control close count 1/);
        const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.exitCode, 0);
        assert.equal(metadata.signal, null);
        assert.equal(metadata.verdict.passed, false);
        assert.match(metadata.verdict.failure, /^Timed out after 2000 ms:/);
      } finally {
        console.error = previousError;
        if (previous === undefined) {
          delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        } else {
          process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previous;
        }
        fs.rmSync(temp, {recursive: true, force: true});
      }
      assert.deepEqual(notices, [
        "Release command deadline reached after 2000 ms; requesting cooperative close; waiting for native cleanup.",
      ]);
    });

  });

  test.describe("Installer security and deployment", () => {
    test("standalone installers authenticate downloaded helper scripts", () => {
      const scripts = path.join(__dirname, "..", "scripts");
      const shellHash = sha256(path.join(scripts, "install-cutlass.sh"));
      const shellInstaller = fs.readFileSync(path.join(scripts, "install.sh"), "utf8");
      assert.match(shellInstaller, new RegExp(`MOM_CUTLASS_HELPER_SHA256=${shellHash}`));
      assert.match(shellInstaller, /sha256sum -c -/);
      assert.match(shellInstaller, /trap "rm -f -- '\$cutlass_helper'" EXIT/);
      assert.match(shellInstaller, /mo-miner\/v0\.9\.0\/scripts\/install-cutlass\.sh/);
      assert.doesNotMatch(shellInstaller, /mo-miner\/master\/scripts\/install-cutlass\.sh/);

      const powershellHash = sha256(path.join(scripts, "install-cutlass.ps1"));
      const powershellInstaller = fs.readFileSync(path.join(scripts, "install.ps1"), "utf8");
      assert.match(powershellInstaller,
        new RegExp(`cutlassHelperSha256 = "${powershellHash}"`, "i"));
      assert.match(powershellInstaller, /Get-FileHash -Algorithm SHA256 \$helper/);
      assert.match(powershellInstaller, /mo-miner\/v0\.9\.0\/scripts\/install-cutlass\.ps1/);
      assert.doesNotMatch(powershellInstaller, /mo-miner\/master\/scripts\/install-cutlass\.ps1/);
      assert.match(powershellInstaller, /function Assert-Authenticode/);
      assert.match(powershellInstaller,
        /Invoke-WebRequest[^\n]*\$fallbackVsBuildToolsUrl[\s\S]*?Assert-Authenticode \$vsInstaller/);
      assert.match(powershellInstaller,
        /Invoke-WebRequest[^\n]*\$url[\s\S]*?Assert-Authenticode \$cudaInstaller/);
      assert.match(powershellInstaller,
        /Invoke-WebRequest[^\n]*\$openClCpuUrl[\s\S]*?Assert-Authenticode \$installer/);
      assert.match(powershellInstaller,
        /Get-FileHash -Algorithm SHA256 \$sevenZipMsi[\s\S]*?Install-MsiPackage \$sevenZipMsi/);
      assert.match(powershellInstaller,
        /if \(\$DryRun\)[\s\S]*?exit 0\s*}[\s\S]*?Install-OpenClCpuRuntime/);

      const oneApiInstaller = fs.readFileSync(path.join(__dirname, "..", ".github", "workflows",
        "scripts", "install-oneapi.bat"), "utf8");
      assert.match(oneApiInstaller, /\[Guid\]::NewGuid\(\)\.ToString\(\)/);
      assert.match(oneApiInstaller, /%TEMP%\\mom-oneapi-%RUN_ID%/);
      assert.doesNotMatch(oneApiInstaller, /oneapi_webimage_extracted/);
    });

    test("development installer authenticates artifacts at their acquisition boundary", () => {
      const installer = fs.readFileSync(path.join(__dirname, "..", "scripts", "install-dev.ps1"),
        "utf8");
      const helpers = fs.readFileSync(path.join(__dirname, "..", "scripts",
        "windows-install-helpers.ps1"), "utf8");
      const adaptiveCpp = fs.readFileSync(path.join(__dirname, "..", "scripts",
        "build-windows-adaptivecpp-base.ps1"), "utf8");
      const installMsi = installer.match(/function Install-Msi\([\s\S]*?\n}/)?.[0] ?? "";
      assert.doesNotMatch(installMsi, /Assert-Authenticode/);
      assert.match(installer,
        /Download 'https:\/\/www\.7-zip\.org\/a\/7z2409-x64\.msi' \$sevenZip\s+Assert-Sha256 \$sevenZip \$sevenZipSha256\s+Install-Msi \$sevenZip/);
      assert.match(installer,
        /Download \$hipSdkUrl \$sdkExe\s+Assert-Sha256 \$sdkExe \$hipSdkSha256\s+\$sevenZip/);
      assert.match(adaptiveCpp,
        /Download \$bootstrapUrl \$bootstrapInstaller }\s+Assert-Sha256 \$bootstrapInstaller \$bootstrapSha256\s+Remove-Item \$bootstrapDir/);
      assert.match(installer,
        /Download 'https:\/\/aka\.ms\/vs\/17\/release\/vs_BuildTools\.exe' \$installer\s+Assert-Authenticode \$installer/);
      assert.match(installer,
        /Download \$openClCpuUrl \$installer\s+Assert-Authenticode \$installer/);
      assert.match(installer,
        /Download \$cudaUrl \$installer\s+Assert-Authenticode \$installer/);
      assert.match(helpers,
        /'--retry','5','--retry-all-errors','--retry-delay','5',[\s\S]*?'--continue-at','-','-o',\$OutFile,\$Url/);
      assert.match(installer,
        /if \(-not \$KeepWorkspace\) \{\s+#[^\n]+\n\s+#[^\n]+\n\s+Set-Location -LiteralPath \$repo\s+Assert-NoReparseAncestor \$Workspace/);
    });

    test("host installer preserves an explicit GPU vendor selection across sudo", {
      skip: process.platform === "win32",
    }, () => {
      const installer = path.join(__dirname, "..", "scripts", "install.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-sudo-test-"));
      const bin = path.join(temp, "bin");
      const argsPath = path.join(temp, "sudo-args");
      fs.mkdirSync(bin);
      fs.writeFileSync(path.join(bin, "id"), [
        "#!/bin/sh",
        "if [ \"$1\" = -u ]; then printf '%s\\n' 1000; else exit 1; fi",
      ].join("\n"));
      fs.writeFileSync(path.join(bin, "sudo"), [
        "#!/bin/sh",
        "printf '%s\\n' \"$@\" > \"$MOM_SUDO_ARGS\"",
        "exit 77",
      ].join("\n"));
      fs.chmodSync(path.join(bin, "id"), 0o755);
      fs.chmodSync(path.join(bin, "sudo"), 0o755);
      /** @param {Record<string, string>} configured */
      const run = (configured) => {
        fs.rmSync(argsPath, {force: true});
        /** @type {TestEnvironment} */
        const env = {...process.env, MOM_SUDO_ARGS: argsPath,
          PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`};
        for (const name of ["MOM_INSTALL_GPU_VENDORS", "ROCM_PATH", "HIP_PATH"]) {
          delete env[name];
        }
        Object.assign(env, configured);
        const result = spawnSync("bash", [installer, "--fixture-argument"], {
          encoding: "utf8", env,
        });
        assert.equal(result.status, 77, `${result.stdout}\n${result.stderr}`);
        return fs.readFileSync(argsPath, "utf8").trim().split(/\r?\n/);
      };
      try {
        assert.deepEqual(run({MOM_INSTALL_GPU_VENDORS: "nvidia"}), [
          "--preserve-env=MOM_INSTALL_GPU_VENDORS", "--", installer, "--fixture-argument",
        ]);
        assert.deepEqual(run({ROCM_PATH: "/fixture/rocm"}), [
          "--preserve-env=ROCM_PATH", "--", installer, "--fixture-argument",
        ]);
        assert.deepEqual(run({HIP_PATH: "/fixture/hip"}), [
          "--preserve-env=HIP_PATH", "--", installer, "--fixture-argument",
        ]);
        assert.deepEqual(run({
          MOM_INSTALL_GPU_VENDORS: "nvidia", ROCM_PATH: "/fixture/rocm", HIP_PATH: "/fixture/hip",
        }), [
          "--preserve-env=MOM_INSTALL_GPU_VENDORS,ROCM_PATH,HIP_PATH", "--", installer,
          "--fixture-argument",
        ]);
        assert.deepEqual(run({}), ["--", installer, "--fixture-argument"]);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("host installer adds the sudo user only to missing GPU access groups", {
      skip: process.platform === "win32",
    }, () => {
      const installer = path.join(__dirname, "..", "scripts", "install.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-groups-test-"));
      const bin = path.join(temp, "bin");
      const usermodLog = path.join(temp, "usermod.log");
      fs.mkdirSync(bin);
      fs.writeFileSync(path.join(bin, "id"), [
        "#!/bin/sh",
        "if [ \"$1\" = -u ] && [ \"$#\" = 1 ]; then printf '%s\\n' 0; exit 0; fi",
        "if [ \"$1\" = -u ] && [ \"$3\" = fixture-user ]; then printf '%s\\n' 1000; exit 0; fi",
        "if [ \"$1\" = -nG ] && [ \"$3\" = fixture-user ]; then printf '%s\\n' \"$MOM_FIXTURE_GROUPS\"; exit 0; fi",
        "exit 1",
      ].join("\n"), {mode: 0o755});
      fs.writeFileSync(path.join(bin, "getent"), [
        "#!/bin/sh",
        "[ \"$1\" = group ] && { [ \"$2\" = render ] || [ \"$2\" = video ]; }",
      ].join("\n"), {mode: 0o755});
      fs.writeFileSync(path.join(bin, "usermod"), [
        "#!/bin/sh",
        "printf '%s\\n' \"$*\" > \"$MOM_USERMOD_LOG\"",
      ].join("\n"), {mode: 0o755});
      for (const command of ["apt-get", "ldconfig"]) {
        fs.writeFileSync(path.join(bin, command), "#!/bin/sh\nexit 0\n", {mode: 0o755});
      }
      /** @type {TestEnvironment} */
      const env = {
        ...process.env,
        MOM_INSTALL_GPU_VENDORS: "opencl",
        MOM_FIXTURE_GROUPS: "fixture-user video",
        MOM_USERMOD_LOG: usermodLog,
        SUDO_UID: "1000",
        SUDO_USER: "fixture-user",
        PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
      };
      try {
        const result = spawnSync("bash", [installer], {encoding: "utf8", env});
        assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
        assert.equal(fs.readFileSync(usermodLog, "utf8").trim(), "-aG render -- fixture-user");
        assert.match(result.stdout, /Added fixture-user to the GPU access group\(s\): render/);
        assert.match(result.stdout, /Sign out and back in/);

        fs.rmSync(usermodLog);
        const complete = spawnSync("bash", [installer], {encoding: "utf8", env: {
          ...env, MOM_FIXTURE_GROUPS: "fixture-user render video",
        }});
        assert.equal(complete.status, 0, `${complete.stdout}\n${complete.stderr}`);
        assert.equal(fs.existsSync(usermodLog), false);
        assert.doesNotMatch(complete.stdout, /Added fixture-user/);

        const directRootEnv = {...env};
        delete directRootEnv["SUDO_UID"];
        delete directRootEnv["SUDO_USER"];
        const directRoot = spawnSync("bash", [installer], {encoding: "utf8", env: directRootEnv});
        assert.equal(directRoot.status, 0, `${directRoot.stdout}\n${directRoot.stderr}`);
        assert.equal(fs.existsSync(usermodLog), false);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("host installer does not add distro ROCm beside a coherent vendor core", {
      skip: process.platform === "win32",
    }, () => {
      const sourceInstaller = path.join(__dirname, "..", "scripts", "install.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-rocm-test-"));
      const installer = path.join(temp, "install.sh");
      const bin = path.join(temp, "bin");
      const aptLog = path.join(temp, "apt.log");
      const rocm = path.join(temp, "rocm", "core-7.14");
      const source = fs.readFileSync(sourceInstaller, "utf8");
      const defaultRoot = "  roots+=(/opt/rocm)";
      assert.equal(source.split(defaultRoot).length, 2);
      // Build containers can own a real /opt/rocm. Keep both fixture branches independent of it.
      fs.writeFileSync(installer, source.replace(defaultRoot,
        `  roots+=(${JSON.stringify(path.join(temp, "system-rocm"))})`));
      fs.mkdirSync(bin, {recursive: true});
      fs.mkdirSync(path.join(rocm, "bin"), {recursive: true});
      fs.mkdirSync(path.join(rocm, "lib"));
      fs.writeFileSync(path.join(rocm, "bin", "hipconfig"), "#!/bin/sh\n", {mode: 0o755});
      fs.symlinkSync(path.join(rocm, "bin", "hipconfig"), path.join(bin, "hipconfig"));
      for (const library of [
        "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
        "libhiprtc.so.7", "libhiprtc-builtins.so.7",
      ]) {
        fs.writeFileSync(path.join(rocm, "lib", library), "test");
      }
      fs.writeFileSync(path.join(bin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n", {mode: 0o755});
      fs.writeFileSync(path.join(bin, "apt-get"), [
        "#!/bin/sh",
        "printf '%s\\n' \"$*\" >> \"$MOM_APT_LOG\"",
      ].join("\n"), {mode: 0o755});
      for (const command of ["dpkg-query", "ldconfig"]) {
        fs.writeFileSync(path.join(bin, command), "#!/bin/sh\nexit 0\n", {mode: 0o755});
      }
      /** @type {NodeJS.ProcessEnv} */
      const env = {
        ...process.env,
        MOM_APT_LOG: aptLog,
        MOM_INSTALL_GPU_VENDORS: "amd",
        PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
      };
      delete env["ROCM_PATH"];
      delete env["HIP_PATH"];
      try {
        const preserved = spawnSync("bash", [installer], {encoding: "utf8", env});
        assert.equal(preserved.status, 0, `${preserved.stdout}\n${preserved.stderr}`);
        assert.match(preserved.stdout, /keeping the coherent ROCm core .*core-7\.14\/lib/);
        const preservedApt = fs.readFileSync(aptLog, "utf8");
        assert.match(preservedApt, /install -y --no-install-recommends libomp5/);
        assert.doesNotMatch(preservedApt, /libamdhip64/);

        fs.rmSync(path.join(rocm, "lib", "libhiprtc-builtins.so.7"));
        fs.rmSync(aptLog);
        const installed = spawnSync("bash", [installer], {encoding: "utf8", env});
        assert.equal(installed.status, 0, `${installed.stdout}\n${installed.stderr}`);
        assert.match(fs.readFileSync(aptLog, "utf8"),
          /libamdhip64-7 libamdhip64-dev libhiprtc7 libhiprtc-builtins7/);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("host installer preserves complete CUDA roots and rejects unsafe roots before download", {
      skip: process.platform === "win32",
    }, () => {
      const sourceInstaller = path.join(__dirname, "..", "scripts", "install.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-cuda-test-"));
      const installer = path.join(temp, "install.sh");
      const bin = path.join(temp, "bin");
      const aptLog = path.join(temp, "apt.log");
      const cudaRoot = path.join(temp, "usr-local-cuda");
      const cudaTarget = path.join(temp, "cuda-target");
      const payloadRoot = path.join(temp, "nvidia-cuda-ubuntu");
      const ldConfig = path.join(temp, "mom-cuda.conf");
      const source = fs.readFileSync(sourceInstaller, "utf8");
      /** @type {Array<[string, string]>} */
      const rewrites = [
        ["/usr/local/cuda", cudaRoot],
        ["/opt/nvidia-cuda-ubuntu", payloadRoot],
        ["/etc/ld.so.conf.d/mom-cuda.conf", ldConfig],
      ];
      let isolated = source;
      for (const [realPath, fixturePath] of rewrites) {
        assert.ok(source.includes(realPath), `source path missing: ${realPath}`);
        isolated = isolated.replaceAll(realPath, fixturePath);
        assert.equal(isolated.includes(realPath), false, `real path leaked: ${realPath}`);
      }
      fs.writeFileSync(installer, isolated, {mode: 0o755});
      fs.writeFileSync(path.join(temp, "install-cutlass.sh"),
        "install_cutlass_headers() { :; }\n");
      fs.mkdirSync(bin);
      fs.writeFileSync(path.join(bin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n", {mode: 0o755});
      fs.writeFileSync(path.join(bin, "ldconfig"), [
        "#!/bin/sh",
        "if [ \"$1\" = -p ]; then printf '%s\\n' libcuda.so.1; fi",
      ].join("\n"), {mode: 0o755});
      fs.writeFileSync(path.join(bin, "apt-cache"), [
        "#!/bin/sh",
        "printf '%s\\n' \"apt-cache $*\" >> \"$MOM_APT_LOG\"",
        "exit 0",
      ].join("\n"), {mode: 0o755});
      fs.writeFileSync(path.join(bin, "apt-get"), [
        "#!/bin/sh",
        "printf '%s\\n' \"$*\" >> \"$MOM_APT_LOG\"",
        "for arg in \"$@\"; do",
        "  if [ \"$arg\" = download ]; then exit 23; fi",
        "done",
        "exit 0",
      ].join("\n"), {mode: 0o755});
      /** @type {TestEnvironment} */
      const env = {
        ...process.env,
        MOM_APT_LOG: aptLog,
        MOM_INSTALL_GPU_VENDORS: "nvidia",
        PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
      };
      const run = () => spawnSync("bash", [installer], {encoding: "utf8", env});
      const resetLog = () => fs.rmSync(aptLog, {force: true});
      const assertNoToolkitDownload = () => {
        const log = fs.readFileSync(aptLog, "utf8");
        assert.doesNotMatch(log, /\bdownload\b/);
        assert.doesNotMatch(log, /apt-cache/);
      };
      const ptxas = path.join(cudaTarget, "bin", "ptxas");
      const include = path.join(cudaTarget, "include");
      /** @type {[string, string, string, string]} */
      const headers = [
        path.join(include, "cuda.h"),
        path.join(include, "cuda_runtime.h"),
        path.join(include, "nvrtc.h"),
        path.join(include, "cuda", "std", "cstdint"),
      ];
      const libdevice = path.join(cudaTarget, "nvvm", "libdevice", "libdevice.10.bc");
      /** @type {[string, string]} */
      const libraries = [
        path.join(cudaTarget, "lib64", "libnvrtc.so.12"),
        path.join(cudaTarget, "lib64", "libnvrtc-builtins.so.12"),
      ];
      try {
        fs.mkdirSync(path.dirname(ptxas), {recursive: true});
        fs.mkdirSync(include, {recursive: true});
        fs.mkdirSync(path.dirname(headers[3]), {recursive: true});
        fs.mkdirSync(path.dirname(libdevice), {recursive: true});
        fs.mkdirSync(path.dirname(libraries[0]), {recursive: true});
        fs.writeFileSync(ptxas, "ptxas fixture");
        fs.chmodSync(ptxas, 0o755);
        for (const header of headers) {
          fs.writeFileSync(header, "CUDA fixture");
        }
        fs.writeFileSync(libdevice, "libdevice fixture");
        for (const library of libraries) {
          fs.writeFileSync(library, "NVRTC fixture");
        }
        fs.symlinkSync(cudaTarget, cudaRoot, "dir");
        const linkTarget = fs.readlinkSync(cudaRoot);
        /** @type {Map<string, Buffer>} */
        const sdkBytes = new Map();
        for (const file of [ptxas, ...headers, libdevice, ...libraries]) {
          sdkBytes.set(file, fs.readFileSync(file));
        }

        const complete = run();
        assert.equal(complete.status, 0, `${complete.stdout}\n${complete.stderr}`);
        assert.match(complete.stdout, /Keeping the complete CUDA source-JIT toolkit/);
        assert.doesNotMatch(complete.stdout, /Installing the NVIDIA source-JIT toolchain/);
        assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
        assert.equal(fs.readlinkSync(cudaRoot), linkTarget);
        for (const [file, bytes] of sdkBytes) {
          assert.deepEqual(fs.readFileSync(file), bytes);
        }
        assert.equal(fs.existsSync(payloadRoot), false);
        assert.equal(fs.existsSync(ldConfig), false);
        assertNoToolkitDownload();

        fs.rmSync(headers[2]);
        resetLog();
        const missing = run();
        assert.notEqual(missing.status, 0, `${missing.stdout}\n${missing.stderr}`);
        assert.match(missing.stderr, /repair the existing CUDA installation/);
        assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
        assert.equal(fs.readlinkSync(cudaRoot), linkTarget);
        assert.equal(fs.existsSync(payloadRoot), false);
        assertNoToolkitDownload();

        fs.writeFileSync(headers[2], "CUDA fixture");
        fs.rmSync(cudaRoot);
        const danglingTarget = path.join(temp, "missing-cuda-target");
        fs.symlinkSync(danglingTarget, cudaRoot, "dir");
        resetLog();
        const dangling = run();
        assert.notEqual(dangling.status, 0, `${dangling.stdout}\n${dangling.stderr}`);
        assert.match(dangling.stderr, /repair the existing CUDA installation/);
        assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
        assert.equal(fs.readlinkSync(cudaRoot), danglingTarget);
        assert.equal(fs.existsSync(payloadRoot), false);
        assertNoToolkitDownload();

        fs.rmSync(cudaRoot);
        resetLog();
        const absent = run();
        assert.notEqual(absent.status, 0, `${absent.stdout}\n${absent.stderr}`);
        assert.match(absent.stdout, /Installing the NVIDIA source-JIT toolchain/);
        assert.match(fs.readFileSync(aptLog, "utf8"), /\bdownload\b/);
        assert.equal(fs.existsSync(cudaRoot), false);
        assert.equal(fs.existsSync(payloadRoot), false);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("Linux development installer cleans only dedicated workspace names", {
      skip: process.platform !== "linux",
    }, () => {
      const scripts = path.join(__dirname, "..", "scripts");
      const linux = path.join(scripts, "install-dev.sh");
      for (const workspace of ["/", "/tmp/compiler-workspace"]) {
        const result = spawnSync("bash", [linux, "--component", "base", "--validate-only",
          "--workspace", workspace], {encoding: "utf8"});
        assert.equal(result.status, 2, result.stderr);
        assert.match(result.stderr, /dedicated mom-dev-/);
      }
      const validationParent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dev-validation-test-"));
      const validationWorkspace = path.join(validationParent, "mom-dev-validation");
      const validationBin = path.join(validationParent, "bin");
      fs.mkdirSync(validationBin);
      fs.writeFileSync(path.join(validationBin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n");
      fs.chmodSync(path.join(validationBin, "id"), 0o755);
      const validationEnv = {
        ...process.env,
        PATH: `${validationBin}${path.delimiter}${process.env["PATH"] ?? ""}`,
      };
      try {
        const validation = spawnSync("bash", [linux, "--component", "base", "--validate-only",
          "--workspace", validationWorkspace], {encoding: "utf8", env: validationEnv});
        assert.equal(validation.error, undefined, validation.error?.message);
        assert.equal(validation.status, 0, validation.stderr);
        assert.equal(fs.existsSync(validationWorkspace), false);

        const wrongMarkerWorkspace = path.join(validationParent, "mom-dev-wrong-marker");
        const wrongMarker = path.join(wrongMarkerWorkspace, ".mom-dev-workspace");
        const sentinel = path.join(wrongMarkerWorkspace, "sentinel");
        fs.mkdirSync(wrongMarkerWorkspace);
        fs.writeFileSync(wrongMarker, "wrong marker");
        fs.writeFileSync(sentinel, "keep");
        const wrongMarkerResult = spawnSync("bash", [linux, "--component", "base",
          "--workspace", wrongMarkerWorkspace], {encoding: "utf8", env: validationEnv});
        assert.equal(wrongMarkerResult.status, 2, wrongMarkerResult.stderr);
        assert.match(wrongMarkerResult.stderr, /\.mom-dev-workspace marker/);
        assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
      } finally {
        fs.rmSync(validationParent, {recursive: true, force: true});
      }
    });

    test("Windows development installer requires dedicated owned workspaces", () => {
      const windows = fs.readFileSync(path.join(__dirname, "..", "scripts", "install-dev.ps1"), "utf8");
      assert.match(windows, /\$workspaceLeaf -notmatch '\^mom-dev-/);
      assert.match(windows, /\$workspaceMarker = Join-Path \$Workspace '\.mom-dev-workspace'/);
      assert.match(windows, /\[IO\.FileAttributes\]::ReparsePoint/);
      assert.match(windows, /Test-MomMarker \$workspaceMarker 'mom development workspace'/);
      assert.match(windows, /Remove-Item -LiteralPath \$Workspace -Recurse/);
    });

    test("ordinary and reused development workers prefer the matching portable runtime", {
      skip: process.platform === "win32",
    }, () => {
      const source = fs.readFileSync(path.join(__dirname, "..", "scripts", "multicompiler-entrypoint.sh"), "utf8");
      const setups = [...source.matchAll(/^[ ]*if [^\n]+\n[ ]+export PATH="\/opt\/dpcpp\/bin:[^\n]+\n[ ]+export LD_LIBRARY_PATH=[^\n]+\n[ ]*fi/gm)];
      assert.equal(setups.length, 2, "both ordinary and reuse setup paths must be tested");
      for (const setup of setups) {
        for (const backend of ["intel_igpu", "intel", "opencl", "nvidia"]) {
          const result = spawnSync("bash", ["-c", setup[0] +
            '\nprintf "%s\\n%s\\n%s\\n%s\\n" "$PATH" "$LD_LIBRARY_PATH" "$OCL_ICD_FILENAMES" "$ONEAPI_DEVICE_SELECTOR"'], {
            encoding: "utf8",
            env: {PATH: "/usr/bin:/bin", LD_LIBRARY_PATH: "/oneapi/lib", backend,
              OCL_ICD_FILENAMES: "existing-icd", ONEAPI_DEVICE_SELECTOR: "level_zero:gpu"},
          });
          assert.equal(result.status, 0, result.stderr);
          const portable = backend !== "intel";
          assert.deepEqual(result.stdout.trimEnd().split("\n"), [
            portable ? "/opt/dpcpp/bin:/usr/bin:/bin" : "/usr/bin:/bin",
            portable ? "/opt/dpcpp/lib:/oneapi/lib" : "/oneapi/lib",
            "existing-icd", "level_zero:gpu",
          ], backend);
        }
      }
    });

    test("combined Dockerfile uses canonical installer defaults", () => {
      const scripts = path.join(__dirname, "..", "scripts");
      const dockerfile = fs.readFileSync(path.join(scripts, "build-combined.dockerfile"), "utf8");
      const intelRuntime = fs.readFileSync(path.join(scripts, "install-intel-compute-runtime.sh"),
        "utf8");
      assert.match(dockerfile,
        /RUN bash \/tmp\/mom-install-dev\/install-dev\.sh \\\n {6}--component base,node,oneapi,cuda,dpcpp/);
      assert.match(dockerfile, /COPY scripts\/install-intel-compute-runtime\.sh/);
      assert.match(dockerfile, /RUN bash \/tmp\/mom-install-dev\/install-intel-compute-runtime\.sh/);
      assert.match(intelRuntime, /readonly COMPUTE_RUNTIME_VERSION=26\.31\.39395\.13/);
      assert.match(intelRuntime, /readonly IGC_VERSION=2\.40\.13/);
      assert.match(intelRuntime, /readonly LEVEL_ZERO_VERSION=1\.32\.0/);
      assert.match(intelRuntime, /github\.com\/intel\/compute-runtime\/releases\/download/);
      assert.match(intelRuntime, /github\.com\/intel\/intel-graphics-compiler\/releases\/download/);
      assert.match(intelRuntime, /github\.com\/oneapi-src\/level-zero\/releases\/download/);
      assert.match(intelRuntime, /ocl-icd-libopencl1 "\$\{packages\[@\]\}"/);
      assert.equal((intelRuntime.match(/[0-9a-f]{64}/g) ?? []).length, 8);
      assert.doesNotMatch(intelRuntime, /\/latest(?:\/|\b)/);
      assert.match(dockerfile,
        /export MOM_AUTOLYKOS2_INTENSITY="\$\{MOM_AUTOLYKOS2_INTENSITY:-\}"/);
      assert.match(dockerfile,
        /for v in MOM_CN_GPU_INTENSITY MOM_NEXAPOW_INTENSITY MOM_AUTOLYKOS2_INTENSITY MOM_AUTOLYKOS2_WORKGROUP MOM_AUTOLYKOS2_SPLIT MOM_AUTOLYKOS2_PROFILE ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK; do\\n/);
      assert.doesNotMatch(dockerfile, /^ARG (?:NODE_VERSION|DPCPP_RELEASE|DPCPP_ASSET)=/m);
      assert.doesNotMatch(dockerfile, /MOM_(?:NODE_VERSION|DPCPP_RELEASE|DPCPP_ASSET)\s*=/);
    });

    test("Windows multicompiler release contracts stay synchronized", () => {
      const root = path.join(__dirname, "..");
      /** @param {...string} parts */
      const read = (...parts) => fs.readFileSync(path.join(root, ...parts), "utf8");
      const amdBuilder = read("scripts", "build-windows-adaptivecpp-amd.ps1");
      assert.match(amdBuilder,
        /\. \(Join-Path \$PSScriptRoot 'windows-install-helpers\.ps1'\)\n\. \(Join-Path \$PSScriptRoot 'import-vcvars\.ps1'\)/);
      assert.match(amdBuilder, /\nImport-MomVcVars64\n/);
      assert.doesNotMatch(amdBuilder, /\$vswhere|\$vsRoot|\$vcvars|\$vcenv/);
      assert.match(amdBuilder,
        /Remove-Item -LiteralPath \$acpp, \$source, \$build -Recurse -Force/);

      const currentTest = read("scripts", "test-windows-current-multicompiler.ps1");
      assert.match(currentTest, /\[ValidateSet\('intel','intel_igpu','nvidia','amd','opencl'\)\]/);
      assert.match(currentTest,
        /'opencl' \{\n {4}if \(\$Compiler -in @\('all','portable'\)\) \{\n {6}Invoke-PortableSuite 'OpenCL'\n {6}\$portableLanes = 1\n {4}\}\n {2}\}/);

      const workflow = read(".github", "workflows", "build-release-artifacts.yml");
      const baseHash = "hashFiles('scripts/build-windows-adaptivecpp-base.ps1', "
    + "'scripts/windows-install-helpers.ps1', 'scripts/import-vcvars.ps1')";
      const overlayHash = "hashFiles('scripts/install-dev.ps1', "
    + "'scripts/build-windows-adaptivecpp-base.ps1', "
    + "'scripts/build-windows-adaptivecpp-amd.ps1', "
    + "'scripts/windows-install-helpers.ps1', 'scripts/import-vcvars.ps1')";
      assert.equal(workflow.split(baseHash).length - 1, 3);
      assert.equal(workflow.split(overlayHash).length - 1, 2);

      const packageWindows = read(".github", "workflows", "scripts", "package-windows.ps1");
      assert.doesNotMatch(packageWindows, /ONEAPI_DEVICE_SELECTOR=level_zero:%MOM_GPU_INDEX%/);
      assert.match(packageWindows, /addon applies MOM_GPU_INDEX after stable hardware-name sorting/);
      assert.match(packageWindows, /ONEAPI_DEVICE_SELECTOR=level_zero:gpu/);

      const packageLinux = read(".github", "workflows", "scripts", "package-linux-combined.sh");
      assert.match(packageLinux, /MOM_GPU_BACKEND=intel\|intel_igpu\|nvidia\|amd\|opencl is intentionally explicit/);

      const docs = read("scripts", "windows-multicompiler.md");
      assert.match(docs, /`GPU_GROUP=intel\|arc\|nvidia\|amd` for/);
      assert.match(docs, /MOM_GPU_BACKEND=intel\|intel_igpu\|nvidia\|amd\|opencl/);
      assert.match(docs, /-Backend opencl/);

      const releaseWindows = read(".github", "workflows", "scripts", "test-release-windows.ps1");
      const suiteGuard = 'if ($Suite -notin @("all", "cpu", "gpu", "gpu-discrete", "gpu-integrated", "gpu-multi", "gpu-portable-cpu"))';
      assert.equal(releaseWindows.split(suiteGuard).length - 1, 1);
      assert.match(releaseWindows,
        /\$ProgressPreference = "SilentlyContinue"\nif \(\$Suite -notin @\("all", "cpu", "gpu", "gpu-discrete", "gpu-integrated", "gpu-multi", "gpu-portable-cpu"\)\) \{\n {2}throw "Unknown release test suite: \$Suite"\n\}/);
      const gpuSmokeSelector = "if ($Suite -in @('gpu', 'gpu-discrete', 'gpu-multi') -and\n" +
    "    $null -eq [Environment]::GetEnvironmentVariable('MOM_GPU_BACKEND', 'Process'))";
      assert.equal(releaseWindows.split(gpuSmokeSelector).length - 1, 1);
      const restoreSmokeBackend = releaseWindows.indexOf("if ($restoreUnsetGpuBackendAfterSmoke)");
      const releaseMatrixDispatch = releaseWindows.indexOf("& $node tests/run_hash.js $Suite");
      assert.ok(restoreSmokeBackend > releaseWindows.indexOf(gpuSmokeSelector));
      assert.ok(restoreSmokeBackend < releaseMatrixDispatch);
      assert.match(releaseWindows,
        /\$Suite -eq 'gpu-multi'\) \{ \$env:MOM_REQUIRE_MULTI_GPU_TESTS = '1' \}/);
      assert.match(releaseWindows,
        /\$Suite -eq 'gpu-integrated'\) \{ \$env:MOM_REQUIRE_INTEGRATED_GPU_TESTS = '1' \}/);
      assert.match(releaseWindows, /Windows gpu-integrated release test requires an Intel integrated GPU/);
      assert.match(releaseWindows, /Intel\.\*\\\[integrated\\\]/);
      assert.match(releaseWindows, /Get-CimInstance Win32_VideoController/);
      assert.match(releaseWindows,
        /if \(\$gpuVendors\.Count -ne 1 -or \$gpuVendors\[0\] -ne '8086'\) \{ return \}/);

      const releaseLinux = read(".github", "workflows", "scripts", "test-release-linux.sh");
      assert.match(releaseLinux,
        /all\|cpu\|gpu\|gpu-discrete\|gpu-integrated\|gpu-multi\|gpu-portable-cpu/);
      assert.match(releaseLinux, /MOM_REQUIRE_INTEGRATED_GPU_TESTS=1/);
      assert.match(releaseLinux, /gpu-integrated suite requires an Intel integrated GPU/);
      assert.match(releaseLinux, /Intel\.\*\\\[integrated\\\]/);
      const runHash = read("tests", "run_hash.js");
      assert.match(runHash, /"gpu-integrated": \["tests\/integrated_gpu\.js"\]/);
      assert.match(runHash,
        /MOM_GPU_TEST_VENDORS: portableOpencl \? undefined : configuredVendors/);

      const deploy = read("scripts", "test-deploy.sh");
      assert.match(deploy, /GPU_GROUP=discrete/);
      assert.doesNotMatch(deploy, /GPU_GROUP=all/);
      assert.match(deploy, /MOM_GPU_TEST_VENDORS=nvidia,amd/);
      assert.match(deploy, /-Suite gpu-multi/);
      assert.match(deploy,
        /-v "\$evidence_root:\/mom-perf-evidence"[\s\S]*MOM_RELEASE_PERF_EVIDENCE_DIR=\/mom-perf-evidence/);
      assert.match(deploy,
        /chown -R "\$MOM_RELEASE_PERF_EVIDENCE_UID:\$MOM_RELEASE_PERF_EVIDENCE_GID"/);
      const windowsGpuGate = read("scripts", "test-windows-release-gpu.ps1");
      assert.match(windowsGpuGate, /\$env:MOM_BENCHMARK_GRACEFUL_ONLY = "1"/);
      assert.match(deploy,
        /GPU_GROUP=none WIN_MOM_RUN_BASE="\$WIN_MOM_DEV_BASE" "\$WIN_RUN" \\\n+ {4}--release --download build\/win --download "\$WINDOWS_ARCHIVE"/);
      assert.doesNotMatch(deploy, /GPU_GROUP=nvidia WIN_MOM_RUN_BASE="\$WIN_MOM_DEV_BASE"/);

      const releaseWorkflow = read(".github", "workflows", "release.yml");
      assert.match(releaseWorkflow, /sha256sum "\$filename" > "\$filename\.sha256"/);
      assert.match(releaseWorkflow, /sha256sum -c "\$filename\.sha256"/);
      assert.match(releaseWorkflow, /files=\("\$\{archives\[@\]\}" "\$\{checksums\[@\]\}"\)/);
    });

    test("development installer rejects invalid configuration before privilege escalation", {
      skip: process.platform === "win32",
    }, () => {
      const linux = path.join(__dirname, "..", "scripts", "install-dev.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dev-config-test-"));
      const bin = path.join(temp, "bin");
      const activity = path.join(temp, "activity");
      fs.mkdirSync(bin);
      fs.writeFileSync(path.join(bin, "id"), [
        "#!/bin/sh",
        "printf '%s\\n' id >> \"$MOM_INSTALLER_ACTIVITY\"",
        "printf '%s\\n' 1000",
      ].join("\n"));
      fs.writeFileSync(path.join(bin, "sudo"), [
        "#!/bin/sh",
        "printf '%s\\n' sudo >> \"$MOM_INSTALLER_ACTIVITY\"",
        "exit 99",
      ].join("\n"));
      fs.chmodSync(path.join(bin, "id"), 0o755);
      fs.chmodSync(path.join(bin, "sudo"), 0o755);

      const valid = {
        MOM_NODE_VERSION: "24.15.0",
        MOM_NODE_SHA256: "0".repeat(64),
        MOM_DPCPP_RELEASE: "nightly-2026-07-11",
        MOM_DPCPP_ASSET: "sycl_linux.tar.gz",
        MOM_DPCPP_SHA256: "0".repeat(64),
        MOM_ADAPTIVECPP_COMMIT: "0".repeat(40),
        MOM_CUDA_VERSION: "12-6",
        MOM_ROCM_VERSION: "7.1.1",
        MOM_LLVM_VERSION: "21",
        MOM_BUILD_JOBS: "1",
      };
      /** @type {[string, string][]} */
      const invalid = [
        ["MOM_NODE_VERSION", "bad"],
        ["MOM_NODE_SHA256", "0".repeat(63)],
        ["MOM_DPCPP_RELEASE", ""],
        ["MOM_DPCPP_ASSET", "asset/name"],
        ["MOM_DPCPP_SHA256", "0".repeat(63)],
        ["MOM_ADAPTIVECPP_COMMIT", "0".repeat(39)],
        ["MOM_CUDA_VERSION", "12.6"],
        ["MOM_ROCM_VERSION", "7.1/evil"],
        ["MOM_LLVM_VERSION", "0"],
        ["MOM_BUILD_JOBS", "0"],
      ];
      try {
        for (const [name, value] of invalid) {
          fs.rmSync(activity, {force: true});
          const result = spawnSync("bash", [linux, "--component", "base", "--validate-only"], {
            encoding: "utf8",
            env: {
              ...process.env,
              ...valid,
              [name]: value,
              MOM_INSTALLER_ACTIVITY: activity,
              PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`,
            },
          });
          assert.equal(result.status, 2, `${name}: ${result.stderr}`);
          assert.equal(result.stderr.trim().split(/\r?\n/).length, 1, result.stderr);
          assert.match(result.stderr, new RegExp(name));
          assert.equal(fs.existsSync(activity), false, `${name} triggered activity`);
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("CUTLASS shell installer rejects files and symlinks before download", {
      skip: process.platform === "win32",
    }, () => {
      const script = path.join(__dirname, "..", "scripts", "install-cutlass.sh");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cutlass-safety-test-"));
      const run = (/** @type {string} */ destination, /** @type {string} */ curlCalled) => spawnSync("bash", [
        "-c",
        'source "$1"; curl() { : > "$MOM_CURL_CALLED"; return 97; }; install_cutlass_headers "$2"',
        "mom-cutlass-safety", script, destination,
      ], {encoding: "utf8", env: {...process.env, MOM_CURL_CALLED: curlCalled}});
      try {
        const curlCalled = path.join(temp, "curl-called");
        const file = path.join(temp, "destination-file");
        fs.writeFileSync(file, "sentinel");
        const fileResult = run(file, curlCalled);
        assert.notEqual(fileResult.status, 0);
        assert.match(fileResult.stderr, /not a directory/);
        assert.equal(fs.readFileSync(file, "utf8"), "sentinel");
        assert.equal(fs.existsSync(curlCalled), false);

        const target = path.join(temp, "target");
        const link = path.join(temp, "destination-link");
        fs.mkdirSync(target);
        fs.writeFileSync(path.join(target, "sentinel"), "sentinel");
        fs.symlinkSync(target, link, "dir");
        const linkResult = run(link, curlCalled);
        assert.notEqual(linkResult.status, 0);
        assert.match(linkResult.stderr, /destination symlink/);
        assert.equal(fs.readFileSync(path.join(target, "sentinel"), "utf8"), "sentinel");
        assert.equal(fs.existsSync(curlCalled), false);

        const destination = path.join(temp, "destination");
        const marker = path.join(destination, ".mom-version");
        const sentinel = path.join(temp, "marker-sentinel");
        fs.mkdirSync(destination);
        fs.writeFileSync(sentinel, "keep");
        fs.symlinkSync(sentinel, marker, "file");
        const markerResult = run(destination, curlCalled);
        assert.notEqual(markerResult.status, 0);
        assert.match(markerResult.stderr, /version marker symlink/);
        assert.equal(fs.existsSync(curlCalled), false);
        assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release extractors reject unsafe workspaces and archive paths", {
      skip: process.platform === "win32",
    }, () => {
      const root = path.join(__dirname, "..");
      const archiveRoot = `mom-v${require(path.join(root, "package.json")).version}`;
      const linux = path.join(root, ".github", "workflows", "scripts", "test-release-linux.sh");
      const invalidWorkspace = spawnSync("bash", [linux, "/missing/archive"], {
        encoding: "utf8", env: {...process.env, MOM_RELEASE_TEST_DIR: "/tmp/release-test"},
      });
      assert.equal(invalidWorkspace.status, 2, invalidWorkspace.stderr);
      assert.match(invalidWorkspace.stderr, /dedicated mom-release-/);

      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-archive-test-"));
      try {
        const invalidSuiteWorkspace = path.join(temp, "mom-release-invalid-suite");
        const invalidSuite = spawnSync("bash", [linux, "/missing/archive", "not-a-suite"], {
          encoding: "utf8", env: {...process.env, MOM_RELEASE_TEST_DIR: invalidSuiteWorkspace},
        });
        assert.equal(invalidSuite.status, 2, invalidSuite.stderr);
        assert.match(invalidSuite.stderr, /Unknown release test suite: not-a-suite/);
        assert.equal(fs.existsSync(invalidSuiteWorkspace), false);

        fs.writeFileSync(path.join(temp, "payload"), "unsafe");
        const archive = path.join(temp, "unsafe.tgz");
        const created = spawnSync("tar", ["-czf", archive, "--transform", "s,^payload$,../escape,",
          "-C", temp, "payload"], {encoding: "utf8"});
        assert.equal(created.status, 0, created.stderr);
        const safeArchive = path.join(temp, "safe.tgz");
        const safeRoot = path.join(temp, archiveRoot);
        fs.mkdirSync(safeRoot);
        fs.writeFileSync(path.join(safeRoot, "payload"), "safe");
        const safeCreated = spawnSync("tar", ["-czf", safeArchive, "-C", temp, archiveRoot], {
          encoding: "utf8",
        });
        assert.equal(safeCreated.status, 0, safeCreated.stderr);
        const unrelatedWorkspace = path.join(temp, "mom-release-unrelated");
        const sentinel = path.join(unrelatedWorkspace, "sentinel");
        fs.mkdirSync(unrelatedWorkspace);
        fs.writeFileSync(sentinel, "keep");
        const ownership = spawnSync("bash", [linux, safeArchive], {
          encoding: "utf8", env: {...process.env, MOM_RELEASE_TEST_DIR: unrelatedWorkspace},
        });
        assert.equal(ownership.status, 2, ownership.stderr);
        assert.match(ownership.stderr, /marker|owned/i);
        assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
        const wrongMarkerWorkspace = path.join(temp, "mom-release-wrong-marker");
        const wrongMarkerSentinel = path.join(wrongMarkerWorkspace, "sentinel");
        fs.mkdirSync(wrongMarkerWorkspace);
        fs.writeFileSync(path.join(wrongMarkerWorkspace, ".mom-release-test-workspace"),
          "wrong marker");
        fs.writeFileSync(wrongMarkerSentinel, "keep");
        const wrongMarker = spawnSync("bash", [linux, safeArchive], {
          encoding: "utf8", env: {...process.env, MOM_RELEASE_TEST_DIR: wrongMarkerWorkspace},
        });
        assert.equal(wrongMarker.status, 2, wrongMarker.stderr);
        assert.match(wrongMarker.stderr, /\.mom-release-test-workspace marker/);
        assert.equal(fs.readFileSync(wrongMarkerSentinel, "utf8"), "keep");
        const result = spawnSync("bash", [linux, archive], {
          encoding: "utf8", env: {...process.env,
            MOM_RELEASE_TEST_DIR: path.join(temp, "mom-release-work")},
        });
        assert.notEqual(result.status, 0);
        assert.match(result.stderr, /unsafe path/);
        assert.equal(fs.existsSync(path.join(temp, "escape")), false);

        const outside = path.join(temp, "outside-dir");
        const outsideSentinel = path.join(outside, "sentinel");
        fs.mkdirSync(outside);
        fs.writeFileSync(outsideSentinel, "keep");
        const symlinkRoot = path.join(temp, archiveRoot);
        fs.mkdirSync(symlinkRoot, {recursive: true});
        fs.symlinkSync("../outside-dir", path.join(symlinkRoot, "escape"));
        const symlinkPayload = path.join(temp, "symlink-payload");
        fs.writeFileSync(symlinkPayload, "overwrite");
        const symlinkArchive = path.join(temp, "unsafe-symlink.tgz");
        const symlinkCreated = spawnSync("tar", [
          "-czf", symlinkArchive, "-C", temp, archiveRoot, "symlink-payload",
          "--transform", `s,^symlink-payload$,${archiveRoot}/escape/sentinel,`,
        ], {encoding: "utf8"});
        assert.equal(symlinkCreated.status, 0, symlinkCreated.stderr);
        const symlinkResult = spawnSync("bash", [linux, symlinkArchive], {
          encoding: "utf8", env: {...process.env,
            MOM_RELEASE_TEST_DIR: path.join(temp, "mom-release-symlink-work")},
        });
        assert.notEqual(symlinkResult.status, 0);
        assert.match(symlinkResult.stderr, /unsafe member|symbolic link/i);
        assert.equal(fs.readFileSync(outsideSentinel, "utf8"), "keep");

        fs.rmSync(symlinkRoot, {recursive: true, force: true});
        const packageRoot = path.join(temp, archiveRoot);
        const secondRoot = path.join(temp, "second");
        fs.mkdirSync(packageRoot);
        fs.mkdirSync(secondRoot);
        fs.writeFileSync(path.join(packageRoot, "payload"), "one");
        fs.writeFileSync(path.join(secondRoot, "payload"), "two");

        const duplicateArchive = path.join(temp, "duplicate.tgz");
        const duplicateCreated = spawnSync("tar", [
          "-czf", duplicateArchive, "-C", temp, archiveRoot, archiveRoot,
        ], {encoding: "utf8"});
        assert.equal(duplicateCreated.status, 0, duplicateCreated.stderr);
        const duplicateResult = spawnSync("bash", [linux, duplicateArchive], {
          encoding: "utf8", env: {...process.env,
            MOM_RELEASE_TEST_DIR: path.join(temp, "mom-release-duplicate-work")},
        });
        assert.notEqual(duplicateResult.status, 0);
        assert.match(duplicateResult.stderr, /duplicate member/i);

        const multipleRootsArchive = path.join(temp, "multiple-roots.tgz");
        const multipleRootsCreated = spawnSync("tar", [
          "-czf", multipleRootsArchive, "-C", temp, archiveRoot, "second",
        ], {encoding: "utf8"});
        assert.equal(multipleRootsCreated.status, 0, multipleRootsCreated.stderr);
        const multipleRootsResult = spawnSync("bash", [linux, multipleRootsArchive], {
          encoding: "utf8", env: {...process.env,
            MOM_RELEASE_TEST_DIR: path.join(temp, "mom-release-roots-work")},
        });
        assert.notEqual(multipleRootsResult.status, 0);
        assert.match(multipleRootsResult.stderr, /expected package root/i);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }

      const windows = fs.readFileSync(path.join(root,
        ".github", "workflows", "scripts", "test-release-windows.ps1"), "utf8");
      assert.match(windows, /\$workLeaf -notmatch '\^mom-release-/);
      assert.match(windows, /\$workspaceMarker = Join-Path \$workDir '\.mom-release-test-workspace'/);
      assert.match(windows, /\[IO\.FileAttributes\]::ReparsePoint/);
      assert.match(windows, /Get-Item -LiteralPath \$workDir/);
      assert.match(windows, /Remove-Item -LiteralPath \$workDir -Recurse/);
      assert.match(windows, /Release archive contains an unsafe path/);
      assert.match(windows, /\$expectedRoot = "mom-v\$packageVersion"/);
      assert.match(windows, /HashSet\[string\].*OrdinalIgnoreCase/);
      assert.match(windows, /Release archive contains a duplicate member/);
      assert.match(windows, /Release archive must contain an explicit package root directory/);
      assert.match(windows, /Expand-Archive -LiteralPath/);
    });

    test("Linux deployment packaging forwards custom release version and archive", {
      skip: process.platform === "win32",
    }, () => {
      const sourceRoot = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-args-test-"));
      const scriptRoot = path.join(temp, "scripts");
      const packageScript = path.join(temp, ".github", "workflows", "scripts",
        "package-linux-combined.sh");
      const bin = path.join(temp, "bin");
      const argsPath = path.join(temp, "package-args");
      const dockerArgsPath = path.join(temp, "docker-args");
      const evidenceRoot = path.join(temp, "evidence");
      fs.mkdirSync(scriptRoot, {recursive: true});
      fs.mkdirSync(path.dirname(packageScript), {recursive: true});
      fs.mkdirSync(bin);
      fs.mkdirSync(evidenceRoot);
      fs.copyFileSync(path.join(sourceRoot, "scripts", "test-deploy.sh"),
        path.join(scriptRoot, "test-deploy.sh"));
      fs.writeFileSync(path.join(temp, "package.json"), JSON.stringify({version: "1.0.0",
        scripts: {"build:release": "node scripts/build-release.js"}}));
      fs.copyFileSync(path.join(sourceRoot, "scripts/build-release.js"),
        path.join(scriptRoot, "build-release.js"));
      fs.copyFileSync(path.join(sourceRoot, "scripts/build-environment.js"),
        path.join(scriptRoot, "build-environment.js"));
      // Model the real entrypoint: build once, then execute its supplied command.
      fs.writeFileSync(path.join(temp, "scripts/dev-container.sh"), [
        "#!/bin/sh",
        "export MOM_R_SH=1",
        'printf "%s\\n" "$MOM_PORTABLE_BUILD" "$MOM_GPU_BACKEND" "$MOM_LINUX_BUILD_COMPILER" "$MOM_DOCKER_GPUS" "$MOM_REUSE_BUILT_WORKER" "$MOM_RELEASE_VERSION" "$@" >> "$MOM_BUILD_ARGS"',
        'exec "$@"',
      ].join("\n"));
      fs.writeFileSync(path.join(bin, "docker"), [
        "#!/bin/sh",
        "if [ \"$1\" = run ]; then",
        "  printf '%s\\n' \"$@\" > \"$MOM_DOCKER_ARGS\"",
        "  printf '%s\\n' 'MOM_TEST_SUMMARY 1 1 0 0'",
        "fi",
        "exit 0",
      ].join("\n"));
      // This argument-forwarding test must not depend on the host NVIDIA driver's current binding.
      fs.writeFileSync(path.join(bin, "nvidia-smi"), "#!/bin/sh\nexit 0\n");
      fs.writeFileSync(packageScript, [
        "#!/bin/sh",
        "printf '%s\\n' \"$@\" > \"$MOM_PACKAGE_ARGS\"",
        "exit 0",
      ].join("\n"));
      for (const file of [
        path.join(temp, "scripts/dev-container.sh"), path.join(bin, "docker"), path.join(bin, "nvidia-smi"), packageScript,
      ]) {
        fs.chmodSync(file, 0o755);
      }
      try {
        const result = spawnSync("bash", [path.join(scriptRoot, "test-deploy.sh")], {
          cwd: temp,
          encoding: "utf8",
          env: {
            ...process.env,
            MOM_DEPLOY_TARGET: "linux-nvidia",
            MOM_DEPLOY_SKIP_VECTORS: "1",
            MOM_RELEASE_VERSION: "v9.8.7",
            MOM_LINUX_BUILD_COMPILER: "oneapi",
            MOM_REUSE_BUILT_WORKER: "1",
            MOM_BUILD_ARGS: path.join(temp, "build-args"),
            MOM_PACKAGE_ARGS: argsPath,
            MOM_DOCKER_ARGS: dockerArgsPath,
            MOM_RELEASE_PERF_EVIDENCE_DIR: evidenceRoot,
            PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`,
          },
        });
        assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
        assert.deepEqual(fs.readFileSync(path.join(temp, "build-args"), "utf8").trim().split(/\r?\n/), [
          "1", "all", "all", "0", "0", "9.8.7", "npm", "run", "build:release", "--", "--package-only",
        ]);
        assert.deepEqual(fs.readFileSync(argsPath, "utf8").trim().split(/\r?\n/), [
          "9.8.7",
        ]);
        const dockerArgs = fs.readFileSync(dockerArgsPath, "utf8").trim().split(/\r?\n/);
        assert.ok(dockerArgs.includes("MOM_RELEASE_ARCHIVE=mom-v9.8.7-lin.tgz"));
        assert.ok(dockerArgs.includes(
          `${path.join(evidenceRoot, "nvidia-linux")}:/mom-perf-evidence`));
        assert.ok(dockerArgs.includes("MOM_RELEASE_PERF_EVIDENCE_DIR=/mom-perf-evidence"));
        assert.match(dockerArgs.join("\n"), /export MOM_BENCHMARK_GRACEFUL_ONLY=1/);
        const user = os.userInfo();
        assert.ok(dockerArgs.includes(`MOM_RELEASE_PERF_EVIDENCE_UID=${user.uid}`));
        assert.ok(dockerArgs.includes(`MOM_RELEASE_PERF_EVIDENCE_GID=${user.gid}`));
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("dev container binds its repository and forwards npm arguments while preserving cache and stdin", {
      skip: process.platform === "win32",
    }, () => {
      const sourceRoot = path.join(__dirname, "..");
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-r-sh-cache-test-"));
      const bin = path.join(temp, "bin");
      const copiedRsh = path.join(temp, "scripts/dev-container.sh");
      const volume = "type=volume,source=mom-sycl-cache-r-sh-test,target=/root/.cache/libsycl_cache";
      fs.mkdirSync(bin);
      fs.mkdirSync(path.join(temp, "home"));
      fs.mkdirSync(path.dirname(copiedRsh), {recursive: true});
      fs.copyFileSync(path.join(sourceRoot, "scripts/dev-container.sh"), copiedRsh);
      for (const file of ["dev.js", "build-environment.js", "windows-command.js"]) {
        fs.copyFileSync(path.join(sourceRoot, "scripts", file), path.join(temp, "scripts", file));
      }
      fs.chmodSync(copiedRsh, 0o755);
      const packageJson = JSON.parse(fs.readFileSync(path.join(sourceRoot, "package.json"), "utf8"));
      fs.writeFileSync(path.join(temp, "package.json"), JSON.stringify({scripts: {dev: packageJson.scripts.dev}}));
      const forwarded = ["node", "argument with spaces", "--", "literal;$value*"];

      fs.writeFileSync(path.join(bin, "docker"), [
        "#!/usr/bin/env bash",
        "set -eu",
        "case \"$1 $2\" in",
        "  \"buildx version\"|\"image inspect\") exit 0 ;;",
        "  \"container inspect\") exit 1 ;;",
        "  \"run \"*)",
        "    shift",
        "    printf '%s\\n' \"$@\" >\"$R_SH_CAPTURE\"",
        "    exit 0",
        "    ;;",
        "  *) exit 97 ;;",
        "esac",
        "",
      ].join("\n"));
      fs.chmodSync(path.join(bin, "docker"), 0o755);
      const initialized = spawnSync("git", ["init", "-q", temp], {encoding: "utf8"});
      assert.equal(initialized.error, undefined, initialized.error?.message);
      assert.equal(initialized.status, 0, initialized.stderr);

      const baseEnv = {
        PATH: bin + ":/usr/bin:/bin",
        HOME: path.join(temp, "home"),
        MOM_GPU_BACKEND: "intel",
        MOM_REUSE_BUILT_WORKER: "1",
        MOM_DOCKER_GPUS: "0",
        MOM_CONTAINER_NETWORK: "none",
        MOM_CONTAINER_CPUS: "1",
        MOM_CONTAINER_MEMORY: "256m",
        MOM_CONTAINER_PIDS: "64",
        MOM_GPU_TEST_VENDORS: "intel",
        MOM_SYCL_CACHE_VOLUME: "mom-sycl-cache-r-sh-test",
      };
      const cases = [
        {name: "default", expected: "SYCL_CACHE_PERSISTENT=1"},
        {name: "explicit-zero", expected: "SYCL_CACHE_PERSISTENT=0", setting: "0"},
        {name: "reuse-stdin", expected: "SYCL_CACHE_PERSISTENT=1", stdin: true},
        {name: "npm-forwarding", expected: "SYCL_CACHE_PERSISTENT=1", npm: true},
        {name: "intel-integrated", expected: "SYCL_CACHE_PERSISTENT=1", integrated: true},
        {name: "intel-integrated-uppercase", expected: "SYCL_CACHE_PERSISTENT=1", integrated: true,
          backend: "INTEL_IGPU"},
      ];
      try {
        for (const [index, testCase] of cases.entries()) {
          const capture = path.join(temp, testCase.name + ".docker-args");
          /** @type {NodeJS.ProcessEnv} */
          const env = {
            ...baseEnv,
            R_SH_CAPTURE: capture,
            MOM_GPU_LOCK_KEY: "cache-test-" + index,
          };
          if (testCase.setting !== undefined) {
            env["SYCL_CACHE_PERSISTENT"] = testCase.setting;
          }
          if (testCase.stdin) {
            env["MOM_CONTAINER_STDIN"] = "1";
          }
          if (testCase.integrated) {
            env["MOM_GPU_BACKEND"] = testCase.backend || "intel_igpu";
            delete env["MOM_GPU_TEST_VENDORS"];
          }
          if (testCase.npm) {
            env["PATH"] = bin + path.delimiter + (process.env["PATH"] || "");
            env["npm_config_update_notifier"] = "false";
            env["npm_config_audit"] = "false";
            env["npm_config_fund"] = "false";
          }
          const command = testCase.npm ? "npm" : "bash";
          const commandArgs = testCase.npm
            ? ["--prefix", temp, "run", "--silent", "dev", "--", ...forwarded]
            : [copiedRsh, ...forwarded];
          const result = spawnSync(command, commandArgs, {
            cwd: os.tmpdir(),
            encoding: "utf8",
            timeout: 10000,
            env,
          });
          assert.equal(result.error, undefined, result.error?.message);
          assert.equal(result.status, 0, testCase.name + ": " + result.stderr);
          const args = fs.readFileSync(capture, "utf8").trimEnd().split(/\r?\n/);
          assert.deepEqual(args.slice(-forwarded.length), forwarded);
          assert.ok(args.some((value) => value.includes(`source=${temp},`)));

          const persistent = args.filter((value) => value.startsWith("SYCL_CACHE_PERSISTENT="));
          assert.deepEqual(persistent, [testCase.expected]);
          const envIndex = args.findIndex((value, position) =>
            value === "--env" && args[position + 1] === testCase.expected);
          assert.notEqual(envIndex, -1);
          const cacheMounts = args.filter((value, position) =>
            value === volume && args[position - 1] === "--mount");
          assert.deepEqual(cacheMounts, [volume]);
          assert.equal(args.filter((value) => value === "--gpus").length, 0);
          assert.equal(args.includes("-i"), testCase.stdin === true);
          if (testCase.integrated) {
            assert.ok(args.includes("MOM_GPU_BACKEND=intel_igpu"));
            assert.ok(args.includes("MOM_GPU_TEST_VENDORS=intel"));
            assert.equal(args.includes("MOM_GPU_TEST_VENDORS=intel_igpu"), false);
          }
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

  });

  test.describe("Release sampling and retained evidence", () => {
    test("release rate parsing supports solution and iteration units", () => {
      const text = [
        "Algo equihash192_7 (gpu1:auto[sycl]) hashrate: 1.5 PH/s",
        "Algo equihash192_7 (gpu1:auto[sycl]) hashrate: 1.2 KH/s",
        "Algo equihash192_7 (gpu1:auto[sycl]) hashrate: 9.16 I/s",
        "Algo equihash192_7 (gpu1:auto[sycl]) hashrate: malformed H/s",
      ].join("\n");
      assert.deepEqual(release.reportedRates(text, "equihash192_7").map((rate) => rate.value),
        [1.5e15, 1200, 9.16]);
      const last = release.lastReportedRate(text, "equihash192_7");
      assert.ok(last);
      assert.equal(last.value, 9.16);
      assert.equal(release.lastReportedRate(
        "Algo equihash192_7 (gpu1:auto[sycl]) hashrate: 0 H/s", "equihash192_7"), null);
    });

    test("release benchmark extends only known cold-start deadlines", () => {
      assert.equal(release.benchmarkTimeout("intel-linux", "nexapow", 300000, 1), 300000);
      assert.equal(release.benchmarkTimeout("nvidia-windows", "nexapow", 300000, 1), 300000);
      assert.equal(release.benchmarkTimeout("intel-windows", "fishhash", 300000, 1), 300000);
      assert.equal(release.benchmarkTimeout("intel-windows", "nexapow", 300000, 1), 1800000);
      assert.equal(release.benchmarkTimeout("intel-windows", "nexapow", 2400000, 1), 2400000);
      assert.equal(release.benchmarkTimeout("amd-windows", "cn/gpu", 300000, 10), 480000);
      assert.equal(release.benchmarkTimeout("amd-linux", "verthash", 300000, 1), 2100000);
      assert.equal(release.benchmarkTimeout("intel-windows", "verthash", 480000, 2), 2280000);
    });

    test("release C30 timeout is fifteen minutes without extending C29", () => {
      assert.equal(release.benchmarkTimeout("intel-linux", "c30", 300000, 1), 900000);
      assert.equal(release.benchmarkTimeout("intel-linux", "c30", 960000, 1), 960000);
      assert.equal(release.benchmarkTimeout("intel-linux", "c29", 300000, 1), 300000);
    });

    test("release benchmark waits through a cold sample before accepting steady rate", async () => {
      const summary = await runReleaseFixture("clean");
      assert.deepEqual(summary, {tests: 1, passed: 1, skipped: 0});
    });

    test("release benchmark writes private per-algorithm evidence", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-evidence-test-"));
      const evidenceRoot = path.join(temp, "evidence");
      try {
        fs.mkdirSync(evidenceRoot, {mode: 0o755});
        const summary = await runReleaseFixture("clean", "fake", "intel-linux", 30000, evidenceRoot);
        assert.deepEqual(summary, {tests: 1, passed: 1, skipped: 0});
        const files = fs.readdirSync(evidenceRoot).sort();
        assert.deepEqual(files, ["fake.metadata.json", "fake.stderr.log", "fake.stdout.log"]);
        if (process.platform !== "win32") {
          assert.equal(fs.statSync(evidenceRoot).mode & 0o777, 0o700);
          for (const file of files) {
            assert.equal(fs.statSync(path.join(evidenceRoot, file)).mode & 0o777, 0o600);
          }
        }
        const metadata = JSON.parse(fs.readFileSync(
          path.join(evidenceRoot, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.schema, "release-performance-benchmark-v3");
        assert.equal(metadata.algo, "fake");
        assert.equal(metadata.platform, "intel-linux");
        assert.equal(new Date(metadata.startedAt).toISOString(), metadata.startedAt);
        assert.equal(new Date(metadata.endedAt).toISOString(), metadata.endedAt);
        assert.ok(Number.isSafeInteger(metadata.elapsedMs) && metadata.elapsedMs >= 0);
        assert.equal(metadata.sampleCount, 4);
        assert.equal(metadata.positiveSampleCount, 4);
        assert.equal(metadata.warmupWindows, 1);
        assert.equal(metadata.steadySamples, 3);
        assert.equal(metadata.requiredSamples, 4);
        assert.match(metadata.stdout, /hashrate:/);
        assert.match(metadata.stderr, /LOOPSTAT/);
        assert.deepEqual(metadata.rates, [80, 100, 100, 100].map((value) => ({
          value, displayValue: value, unit: "H/s",
        })));
        assert.equal(metadata.windowCount, 3);
        assert.deepEqual(metadata.cpuSamples, [
          {timestamp: 1, wall_seconds: 10, cpu_percent: 100, dispatch_percent: 90,
            message_seconds: 0.001, post_seconds: 0.002, iterations: 1, jobs: 1,
            max_message_ms: 0.2, max_post_ms: 0.3},
          {timestamp: 2, wall_seconds: 10, cpu_percent: 10, dispatch_percent: 90,
            message_seconds: 0.001, post_seconds: 0.002, iterations: 1, jobs: 1,
            max_message_ms: 0.2, max_post_ms: 0.3},
          {timestamp: 3, wall_seconds: 10, cpu_percent: 10, dispatch_percent: 90,
            message_seconds: 0.001, post_seconds: 0.002, iterations: 1, jobs: 1,
            max_message_ms: 0.2, max_post_ms: 0.3},
        ]);
        assert.equal(metadata.loopStatsInvalid, false);
        assert.equal(metadata.exitCode, 0);
        assert.equal(metadata.signal, null);
        assert.equal(metadata.gracefulClose, true);
        assert.ok(metadata.stdoutBytes > 0);
        assert.ok(metadata.stderrBytes > 0);
        assert.deepEqual(metadata.verdict, {
          selectedRate: {value: 100, displayValue: 100, unit: "H/s"},
          minimumRate: 95,
          performanceOk: true,
          cpuFailure: null,
          processOk: true,
          passed: true,
          failure: null,
        });
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release benchmark leaves no evidence when the evidence env is unset", async (t) => {
      /** @type {string[]} */
      const mkdirCalls = [];
      const mkdirSync = fs.mkdirSync.bind(fs);
      t.mock.method(fs, "mkdirSync", (/** @type {fs.PathLike} */ target,
        /** @type {fs.MakeDirectoryOptions | undefined} */ options) => {
        mkdirCalls.push(path.resolve(String(target)));
        return mkdirSync(target, options);
      });
      const previousEvidence = process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
      delete process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
      try {
        const summary = await runReleaseFixture("clean");
        assert.deepEqual(summary, {tests: 1, passed: 1, skipped: 0});
        assert.deepEqual(mkdirCalls, []);
      } finally {
        if (previousEvidence === undefined) {
          delete process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
        } else {
          process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"] = previousEvidence;
        }
      }
    });

    for (const phase of ["write", "close"]) {
      test(`release live evidence ${phase} errors fail safely and drain both captures`, async (t) => {
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-live-error-test-"));
        const previousGraceful = process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];
        const open = fs.openSync.bind(fs), write = fs.writeSync.bind(fs), close = fs.closeSync.bind(fs);
        /** @type {Set<number>} */
        const descriptors = new Set();
        let opened = 0, closed = 0;
        t.mock.method(fs, "openSync", (/** @type {fs.PathLike} */ file,
        /** @type {string | number} */ flags, /** @type {string | number | undefined} */ mode) => {
          const fd = open(file, flags, mode);
          if (/fake\.(stdout|stderr)\.log$/.test(String(file))) {descriptors.add(fd); opened++;}
          return fd;
        });
        t.mock.method(fs, "writeSync", (/** @type {number} */ fd,
        /** @type {Uint8Array | string} */ data, /** @type {unknown[]} */ ...options) => {
          if (phase === "write" && descriptors.has(fd)) {throw new Error("fixture live capture write failed");}
          return Reflect.apply(write, fs, [fd, data, ...options]);
        });
        t.mock.method(fs, "closeSync", (/** @type {number} */ fd) => {
          const captured = descriptors.delete(fd);
          if (captured) {closed++;}
          close(fd);
          if (phase === "close" && captured && closed === 1) {throw new Error("fixture live capture close failed");}
        });
        process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = "1";
        try {
          await assert.rejects(runReleaseFixture("clean", "fake", "intel-linux", 30000, temp),
            /fixture live capture (write|close) failed/);
          const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
          assert.equal(metadata.exitCode, 0);
          assert.equal(metadata.signal, null);
          assert.equal(metadata.gracefulClose, true);
          assert.equal(metadata.verdict.passed, false);
          assert.equal(opened, 2);
          assert.equal(closed, 2);
          assert.equal(descriptors.size, 0);
        } finally {
          if (previousGraceful === undefined) {delete process.env["MOM_BENCHMARK_GRACEFUL_ONLY"];} else {
            process.env["MOM_BENCHMARK_GRACEFUL_ONLY"] = previousGraceful;
          }
          fs.rmSync(temp, {recursive: true, force: true});
        }
      });
    }

    test("release live startup capture stays bounded while final metadata keeps diagnostic tails", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-live-bound-test-"));
      try {
        await assert.rejects(runReleaseFixture("compute-error-trimmed", "fake", "intel-linux", 30000, temp),
          /Compute core error was reported/);
        const startup = fs.readFileSync(path.join(temp, "fake.stdout.log"), "utf8");
        assert.equal(fs.statSync(path.join(temp, "fake.stdout.log")).size, performance.maxCapturedOutput);
        assert.match(startup, /hashrate: 80 H\/s/);
        const metadata = JSON.parse(fs.readFileSync(path.join(temp, "fake.metadata.json"), "utf8"));
        assert.ok(metadata.stdout.endsWith("x".repeat(4096)));
        assert.equal(metadata.verdict.passed, false);
      } finally {fs.rmSync(temp, {recursive: true, force: true});}
    });

    test("release performance gate stops promptly on a split compute-core error", async () => {
      const started = Date.now();
      await assert.rejects(
        runReleaseFixture("compute-error", "fake", "intel-linux", 30000),
        /no hashrate was reported[\s\S]*Compute core error/u,
      );
      assert.ok(Date.now() - started < 5000);
    });

    test("release benchmark rejects compute faults before and after its steady window", async () => {
      for (const mode of ["compute-error-after-rate", "compute-error-after-samples",
        "compute-error-trimmed"]) {
        await assert.rejects(runReleaseFixture(mode), /Compute core error was reported/);
      }
    });

    test.describe("Sticky runtime fault evidence", () => {
      for (const mode of ["split", "late", "oversized", "cuda-unloading", "cuda-code"]) {
        test(`release benchmark rejects sticky runtime faults: ${mode}`, async () => {
          const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-runtime-fault-test-"));
          const worker = path.join(temp, "worker.js");
          const evidence = path.join(temp, "evidence");
          fs.writeFileSync(worker, String.raw`
const mode = process.env.MOM_RELEASE_FIXTURE_MODE;
if (process.argv[2] === "algorithms") {
  console.log('gpu1:fixture\nMOM_ALGORITHMS {"fake":"gpu1"}');
} else {
  const fault = mode === "cuda-unloading" ? "cudaErrorCudartUnloading" :
    mode === "cuda-code" ? "error code = CUDA:4" : "[AdaptiveCpp Error] fixture HIP query failed";
  const reportFault = () => {
    const split = Math.floor(fault.length / 2);
    process.stderr.write(fault.slice(0, split), () => setTimeout(() => {
      const padding = mode === "oversized" ? "x".repeat(1024 * 1024 + 4096) : "";
      process.stderr.write(fault.slice(split) + "\n" + padding, () => {
        if (mode !== "split") process.exit(0);
      });
    }, 10));
  };
  process.stdin.on("data", () => {
    if (mode === "split") process.exit(0);
    else reportFault();
  });
  for (let i = 0; i < 3; ++i) {
    console.error("LOOPSTAT t=123 wall=10.0s cpu=4.5% dispatch=99.0% " +
      "msg=0.001s post=0.002s iters=3 jobs=1 max_msg=0.2ms max_post=0.3ms");
  }
  console.log("Algo fake (gpu1) hashrate: 100 H/s\n".repeat(mode === "split" ? 1 : 4));
  if (mode === "split") reportFault();
  setTimeout(() => process.exit(6), 2000).unref();
}
`);
          try {
            await assert.rejects(runReleaseFixture(mode, "fake", "intel-linux", 30000,
              evidence, undefined, worker));
            const metadata = JSON.parse(fs.readFileSync(path.join(evidence, "fake.metadata.json"), "utf8"));
            assert.equal(metadata.exitCode, 0, "fixture must finish through cooperative close");
            assert.equal(metadata.signal, null);
            assert.equal(metadata.gracefulClose, true);
            assert.equal(metadata.verdict.failure, "AdaptiveCpp runtime error was reported");
            assert.equal(metadata.verdict.performanceOk, false);
            assert.equal(metadata.verdict.passed, false);
            if (mode === "oversized") {
              assert.equal(fs.statSync(path.join(evidence, "fake.stderr.log")).size, performance.maxCapturedOutput);
              assert.doesNotMatch(metadata.stderr, /\[AdaptiveCpp Error\]/);
            }
          } finally {fs.rmSync(temp, {recursive: true, force: true});}
        });
      }

      test("release benchmark detects a runtime fault leading one oversized data chunk", async () => {
        const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-runtime-chunk-test-"));
        const readme = path.join(temp, "README.md");
        fs.writeFileSync(readme, "| Algo / coin / pool | GPU | Linux | Windows |\n" +
        "| --- | --- | --- | --- |\n| `fishhash` | 9060 XT | 100 MH/s | 100 MH/s |\n");
        const {EventEmitter} = require("node:events");
        const {PassThrough} = require("node:stream");
        const script = require.resolve("../scripts/check-release-performance");
        const moduleFixture = {exports: /** @type {typeof release} */ ({})};
        const localRequire = require("node:module").createRequire(script);
        /** @param {string} _command @param {string[]} args */
        const spawn = (_command, args) => {
          const child = Object.assign(new EventEmitter(), {
            stdout: new EventEmitter(), stderr: new EventEmitter(), stdin: new PassThrough(),
            exitCode: /** @type {number | null} */ (null), signalCode: null,
          });
          child.stdin.on("finish", () => setImmediate(() => {
          // A single data event exceeds the retained tail, unlike OS pipe-sized chunks.
            child.stderr.emit("data", Buffer.from("[AdaptiveCpp Error] fixture\n" +
            "x".repeat(performance.maxCapturedOutput + 4096)));
            child.exitCode = 0;
            child.emit("close", 0, null);
          }));
          setImmediate(() => {
            if (args.includes("algorithms")) {
              child.stdout.emit("data", Buffer.from('gpu1:fixture\nMOM_ALGORITHMS {"fishhash":"gpu1"}\n'));
              child.exitCode = 0;
              child.emit("close", 0, null);
            } else {
              child.stderr.emit("data", Buffer.from(("LOOPSTAT t=123 wall=10.0s cpu=4.5% dispatch=99.0% " +
              "msg=0.001s post=0.002s iters=3 jobs=1 max_msg=0.2ms max_post=0.3ms\n").repeat(3)));
              child.stdout.emit("data", Buffer.from("Algo fishhash (gpu1) hashrate: 1000 MH/s\n".repeat(4)));
            }
          });
          return child;
        };
        runInNewContext(fs.readFileSync(script, "utf8"), {
          module: moduleFixture,
          require: /** @param {string} name */ name => name === "node:child_process" ? {spawn} : localRequire(name),
          process, Buffer, setTimeout, clearTimeout, console: {log: () => undefined, error: () => undefined},
        });
        const previousEvidence = process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
        delete process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"];
        try {
          const isolated = moduleFixture.exports;
          await assert.rejects(isolated.main(isolated.parseOptions([
            "--miner", releaseFixture, "--readme", readme,
            "--platform", "amd-windows", "--algo", "fishhash",
          ])), /AdaptiveCpp runtime error was reported/);
        } finally {
          if (previousEvidence !== undefined) {process.env["MOM_RELEASE_PERF_EVIDENCE_DIR"] = previousEvidence;}
          fs.rmSync(temp, {recursive: true, force: true});
        }
      });

    });

    test("release benchmark never promotes an incomplete sample window", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-incomplete-test-"));
      try {
        const evidence = path.join(temp, "evidence");
        await assert.rejects(runReleaseFixture("insufficient-samples", "fake", "intel-linux",
          30000, evidence), /benchmark exited with code 0/);
        const metadata = JSON.parse(fs.readFileSync(path.join(evidence, "fake.metadata.json"), "utf8"));
        assert.equal(metadata.sampleCount, 1);
        assert.equal(metadata.requiredSamples, 4);
        assert.equal(metadata.verdict.performanceOk, false);
        assert.equal(metadata.verdict.passed, false);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release benchmark collects compute faults and continues later algorithms", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-fault-matrix-test-"));
      try {
        const evidence = path.join(temp, "evidence");
        await assert.rejects(runReleaseFixture("compute-error-matrix", "fake", "intel-linux",
          30000, evidence, "good"), /Compute core error was reported/);
        const failed = JSON.parse(fs.readFileSync(path.join(evidence, "fake.metadata.json"), "utf8"));
        const passed = JSON.parse(fs.readFileSync(path.join(evidence, "good.metadata.json"), "utf8"));
        assert.equal(failed.verdict.passed, false);
        assert.equal(passed.verdict.passed, true);
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release benchmark evidence does not overwrite existing files", async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-evidence-lock-test-"));
      const evidenceRoot = path.join(temp, "evidence");
      try {
        assert.deepEqual(await runReleaseFixture("clean", "fake", "intel-linux", 30000, evidenceRoot),
          {tests: 1, passed: 1, skipped: 0});
        const files = fs.readdirSync(evidenceRoot).sort();
        const before = new Map(files.map((file) => [file,
          fs.readFileSync(path.join(evidenceRoot, file), "utf8")]));
        await assert.rejects(
          runReleaseFixture("clean", "fake", "intel-linux", 30000, evidenceRoot),
          (error) => error instanceof Error &&
          /** @type {NodeJS.ErrnoException} */ (error).code === "EEXIST");
        assert.deepEqual(fs.readdirSync(evidenceRoot).sort(), files);
        for (const file of files) {
          assert.equal(fs.readFileSync(path.join(evidenceRoot, file), "utf8"), before.get(file));
        }
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release benchmark evidence rejects symlink roots and final files", {
      skip: process.platform === "win32",
    }, async () => {
      const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-evidence-symlink-test-"));
      try {
        const rootTarget = path.join(temp, "root-target");
        const rootLink = path.join(temp, "root-link");
        fs.mkdirSync(rootTarget);
        fs.symlinkSync(rootTarget, rootLink, "dir");
        await assert.rejects(
          runReleaseFixture("clean", "fake", "intel-linux", 30000, rootLink),
          /Evidence root must not be a symlink/);

        const evidenceRoot = path.join(temp, "evidence");
        const fileTarget = path.join(temp, "file-target");
        const fileLink = path.join(evidenceRoot, "fake.metadata.json");
        fs.mkdirSync(evidenceRoot);
        fs.writeFileSync(fileTarget, "existing evidence\n");
        fs.symlinkSync(fileTarget, fileLink);
        await assert.rejects(
          runReleaseFixture("clean", "fake", "intel-linux", 30000, evidenceRoot),
          /Evidence file must not be a symlink/);

        const liveRoot = path.join(temp, "live-evidence");
        fs.mkdirSync(liveRoot);
        fs.symlinkSync(fileTarget, path.join(liveRoot, "fake.stdout.log"));
        await assert.rejects(runReleaseFixture("clean", "fake", "intel-linux", 30000, liveRoot),
          error => error instanceof Error && /** @type {NodeJS.ErrnoException} */ (error).code === "EEXIST");
        assert.equal(fs.readFileSync(fileTarget, "utf8"), "existing evidence\n");
      } finally {
        fs.rmSync(temp, {recursive: true, force: true});
      }
    });

    test("release benchmark evidence uses an exclusive temporary and atomic publication", () => {
      const source = fs.readFileSync(path.join(__dirname, "..", "scripts", "check-release-performance.js"),
        "utf8");
      assert.match(source, /lstatSync\(file\)/);
      assert.match(source, /openSync\(temporary, "wx"/);
      assert.match(source, /fsyncSync\(fd\)/);
      assert.match(source, /linkSync\(temporary, file\)/);
      assert.match(source, /unlinkSync\(temporary\)/);
      assert.match(source, /Evidence root must not be a symlink/);
    });

  });

  test.describe("Algorithm coverage, telemetry, and performance thresholds", () => {
    test("release benchmark passes PearlHash certificate as a dotted option", async () => {
      const summary = await runReleaseFixture("clean", "pearlhash", "nvidia-windows");
      assert.deepEqual(summary, {tests: 1, passed: 1, skipped: 0});
    });

    test("release benchmark includes Verthash without a preexisting dataset", async () => {
      const previous = process.env["MOM_VERTHASH_DATA"];
      delete process.env["MOM_VERTHASH_DATA"];
      try {
        assert.deepEqual(await runReleaseFixture("clean", "verthash"),
          {tests: 1, passed: 1, skipped: 0});
      } finally {
        if (previous !== undefined) {process.env["MOM_VERTHASH_DATA"] = previous;}
      }
    });

    test("Verthash GPU coverage is present without a dataset environment variable", () => {
      const repoRoot = path.join(__dirname, "..");
      const probe = `
    "use strict";
    const assert = require("node:assert/strict");
    const {hashTests} = require(${JSON.stringify(path.join(repoRoot, "tests/vectors.js"))});
    const {gpuAlgos, gpuVectorsFor, fastVectorFor} =
      require(${JSON.stringify(path.join(repoRoot, "tests/common/gpu_test_modes.js"))});
    const verthash = hashTests.filter((definition) => definition.job?.algo === "verthash");
    assert.equal(verthash.length, 1);
    assert.equal(verthash[0].gpu, true);
    assert.equal(verthash[0].syclCpu, true);
    assert.equal(gpuVectorsFor("verthash").length, 1);
    assert.equal(fastVectorFor("verthash"), verthash[0]);
    assert.equal(verthash[0].timeoutMs, 35 * 60 * 1000);
    assert.equal(verthash[0].perfTimeoutMs, 5 * 60 * 1000);
    assert.ok(gpuAlgos.includes("verthash"));
    process.stdout.write("verthash-vector-probe:passed\\n");
  `;
      const env = {...process.env};
      delete env["MOM_VERTHASH_DATA"];
      delete env["NODE_OPTIONS"];
      const result = spawnSync(process.execPath, ["-e", probe], {
        cwd: repoRoot,
        env,
        encoding: "utf8",
      });
      assert.equal(result.status, 0, result.stderr || result.stdout);
      assert.equal(result.signal, null);
      assert.equal(result.stdout.trim(), "verthash-vector-probe:passed");
    });

    test("release benchmark rejects consecutive high CPU telemetry", async () => {
      await assert.rejects(runReleaseFixture("cpu-high"), /CPU telemetry high pair/);
    });

    test("release benchmark accepts an isolated high CPU telemetry spike", async () => {
      assert.deepEqual(await runReleaseFixture("cpu-spike"), {tests: 1, passed: 1, skipped: 0});
    });

    test("release benchmark fails closed when CPU telemetry is missing", async () => {
      await assert.rejects(runReleaseFixture("cpu-missing"), /CPU telemetry missing/);
    });

    test("release benchmark fails closed when CPU telemetry is insufficient", async () => {
      await assert.rejects(runReleaseFixture("cpu-short"), /fewer than two post-setup/);
    });

    test("release benchmark fails closed on invalid CPU telemetry", async () => {
      await assert.rejects(runReleaseFixture("cpu-invalid"), /invalid CPU value/);
    });

    test("release benchmark fails closed on malformed CPU telemetry", async () => {
      await assert.rejects(runReleaseFixture("cpu-malformed"), /invalid LOOPSTAT records/);
    });

    test("release benchmark excludes CPU-assisted cycle solvers from the high CPU guard", async () => {
      for (const algo of ["c29", "c30"]) {
        assert.deepEqual(await runReleaseFixture("cpu-high", algo),
          {tests: 1, passed: 1, skipped: 0});
      }
    });

    test("release benchmark rejects a transient peak before a lower steady rate", async () => {
      await assert.rejects(runReleaseFixture("transient"),
        /94 H\/s is below 95 H\/s; samples: \[100 H\/s, 94 H\/s, 94 H\/s, 94 H\/s\] \(first 1 warm-up\)/);
    });

    test("release benchmark locks the threshold sample before trailing output", async () => {
      await assert.rejects(runReleaseFixture("trailing"), /Release performance regressions/);
    });

    test("release benchmark rejects a crash after a matching rate", async () => {
      await assert.rejects(runReleaseFixture("crash"), /Release performance regressions/);
    });

    test("release benchmark accepts exactly five percent and rejects anything lower", async () => {
      assert.deepEqual(await runReleaseFixture("boundary"), {tests: 1, passed: 1, skipped: 0});
      await assert.rejects(runReleaseFixture("below"), /Release performance regressions/);
    });

    test("release benchmark trims solution-count outliers before averaging", async () => {
      assert.deepEqual(await runReleaseFixture("stochastic", "zhash"),
        {tests: 1, passed: 1, skipped: 0});
    });

    test("release benchmark discards ten cn/gpu warmup windows on Windows", async () => {
      assert.deepEqual(await runReleaseFixture("cn-gpu-ramp", "cn/gpu", "nvidia-windows"),
        {tests: 1, passed: 1, skipped: 0});
    });

    test("performance output capture keeps only a bounded diagnostic tail", () => {
      const tail = performance.appendOutputTail("a".repeat(performance.maxCapturedOutput), "useful-tail");
      assert.equal(tail.length, performance.maxCapturedOutput);
      assert.match(tail, /useful-tail$/);
    });

    test("GPU comparison validates reports and normalizes old samples", () => {
      const baseline = {label: "base", results: [{
        algo: "nexapow", status: "ok", samples: [{value: 2, unit: "KH/s"}],
      }]};
      const candidate = {label: "new", results: [{
        algo: "nexapow", status: "ok", samples: [
          {value: 2.1, unit: "KH/s", value_per_second: 2100},
          {value: 2.3, unit: "KH/s", value_per_second: 2300},
        ],
      }]};
      assert.equal(comparison.median([1, 3, 2, 4]), 2.5);
      assert.equal(comparison.median([1, Number.NaN]), null);
      const baselineResult = baseline.results[0];
      assert.ok(baselineResult);
      const baselineSample = comparison.sampleValue(baselineResult);
      assert.ok(baselineSample);
      assert.equal(baselineSample.normalized, 2000);
      const iterationSample = comparison.sampleValue({algo: "test", samples: [{value: 9.16, unit: "I/s"}]});
      assert.ok(iterationSample);
      assert.equal(iterationSample.normalized, 9.16);
      assert.equal(comparison.sampleValue({algo: "test", samples: [{value: -1, unit: "H/s"}]}), null);
      assert.equal(comparison.sampleValue({algo: "test", samples: [{value: 0, unit: "H/s"}]}), null);
      assert.equal(comparison.sampleValue({algo: "test", samples: [{value: "2", unit: "KH/s"}]}), null);
      assert.match(comparison.renderComparison([baseline, candidate]),
        /\| nexapow \| 2 KH\/s \| 2\.2 KH\/s \(10\.0%\) \|/);
      assert.throws(() => comparison.validateReport({label: "x", results: [
        {algo: "empty", status: "ok", samples: []},
      ]}), /samples are invalid/);
      assert.deepEqual(comparison.validateReport({label: "x", results: [
        {algo: "skipped", status: "not-detected", samples: []},
      ]}).results[0]?.samples, []);
      assert.throws(() => comparison.validateReport({label: "x", results: [
        {algo: "same", status: "ok", samples: [{value: 1, unit: "H/s"}]},
        {algo: "same", status: "ok", samples: [{value: 1, unit: "H/s"}]},
      ]}), /duplicate result algo/);
    });
  });
});
