"use strict";

const {test} = require("node:test");
const assert = require("node:assert/strict");
const {fork, spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {runInNewContext} = require("node:vm");
const helper = require("../helper");

test.describe("Development test runner routing", () => {
  const runnerSource = fs.readFileSync(path.join(__dirname, "common", "miner_command.js"), "utf8");
  const repo = path.resolve(__dirname, "..");

  /** @param {{platform?: string, env?: NodeJS.ProcessEnv, container?: boolean,
   * release?: boolean, devContainer?: boolean}} [options] */
  function runner({platform = "linux", env = {}, container = false,
    release = false, devContainer = true} = {}) {
    const exported = {exports: {}};
    runInNewContext(runnerSource, {
      module: exported, __dirname: path.join(__dirname, "common"),
      process: {platform, env, execPath: "fixture-node"}, setTimeout, clearTimeout,
      /** @param {string} name */
      require(name) {
        if (name === "node:fs") {
          return {existsSync: (/** @type {string} */ file) => file === "/.dockerenv" ? container :
            file === path.join(repo, "scripts", "dev-container.sh") ? devContainer :
              ["mom", "mom.exe", "mom.cmd"].some(leaf => file === path.join(repo, leaf)) && release};
        }
        if (name === "./miner_execution") {return () => ({});}
        if (name === "../../compiler-policy") {return require("../compiler-policy");}
        if (name === "../../scripts/windows-command") {return require("../scripts/windows-command");}
        return require(name);
      },
    });
    return /** @type {typeof import("./common/miner_command")} */ (exported.exports);
  }

  test("only explicit host opt-in changes native source correctness and performance routing", () => {
    for (const script of ["run_hash.js", "run_perf.js"]) {
      for (const marker of [undefined, "", "0", "true", "1"]) {
        const command = runner({env: {MOM_DEV_HOST: marker}});
        /** @type {Array<{command: string, args: string[], env?: TestEnvironment | undefined}>} */
        const calls = [];
        runInNewContext(fs.readFileSync(path.join(__dirname, script), "utf8"), {
          process: {argv: ["fixture-node", script], env: {MOM_DEV_HOST: marker},
            platform: "linux", execPath: "fixture-node"},
          /** @param {string} name */
          require(name) {
            if (name === "./common/miner_command") {
              return {...command, spawnAndExit: (/** @type {string} */ executable,
                /** @type {string[]} */ args, /** @type {{env?: TestEnvironment}} */ options) =>
                calls.push({command: executable, args: Array.from(args), env: options.env})};
            }
            if (name === "node:fs") {return {existsSync: () => true};}
            if (name === "./vectors") {return {perfTests: []};}
            return require(name);
          },
        });
        assert.equal(calls.length, 1);
        assert.equal(calls[0]?.command, marker === "1" ? "fixture-node" : "./scripts/dev-container.sh");
        assert.ok(calls[0]?.args.includes("--test"));
        assert.ok(calls[0]?.args.includes(script === "run_perf.js" ? "tests/perf.js" : "tests/all.js"));
      }
    }
  });

  test("Windows, container, release and legacy Docker defaults remain unchanged", () => {
    for (const options of [{platform: "win32"}, {container: true}, {release: true}, {env: {MOM_R_SH: "1"}}]) {
      assert.equal(runner(options).resolveNodeRunner(["fixture-test.js"]).command, "fixture-node");
    }
    assert.equal(runner({devContainer: false}).resolveNodeRunner(["fixture-test.js"]).command, "./docker-mom.sh");
    assert.equal(runner({devContainer: false, env: {MOM_DEV_HOST: "1"}})
      .resolveNodeRunner(["fixture-test.js"]).command, "fixture-node");
  });
});

test.describe("Helper worker lifecycle", {concurrency: false}, () => {
  const quietHelper = {...helper, log3: () => undefined};

  const fixture = path.join(__dirname, "fixtures", "helper_malformed_worker.js");
  const stdinFixture = path.join(__dirname, "fixtures", "helper_stdin_worker.js");
  const generationFixture = path.join(__dirname, "fixtures", "helper_worker_generation.js");
  const replacementFixture = path.join(__dirname, "fixtures", "helper_worker_replacement.js");
  const shutdownFixture = path.join(__dirname, "fixtures", "helper_shutdown_worker.js");

  /** @param {string | undefined} nativePath @param {() => void} callback */
  function withNativePath(nativePath, callback) {
    const previous = process.env["MOM_NATIVE_PATH"];
    if (nativePath === undefined) {delete process.env["MOM_NATIVE_PATH"];} else {
      process.env["MOM_NATIVE_PATH"] = nativePath;
    }
    try {
      callback();
    } finally {
      if (previous === undefined) {delete process.env["MOM_NATIVE_PATH"];} else {
        process.env["MOM_NATIVE_PATH"] = previous;
      }
    }
  }

  test.describe("Compute addon selection and discovery", {concurrency: false}, () => {
    test("missing explicit compute addon never searches for another worker", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-helper-addon-"));
      const selected = path.join(root, "missing.node");
      const originalExists = fs.existsSync;
      fs.existsSync = () => {throw new Error("Unexpected fallback addon lookup");};
      try {
        withNativePath(selected, () => {
          assert.throws(() => helper.create_core.call(quietHelper), (error) =>
            error instanceof Error && "code" in error && error.code === "MODULE_NOT_FOUND" &&
        error.message.includes(selected));
        });
      } finally {
        fs.existsSync = originalExists;
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("valid explicit compute addon is used without fallback discovery", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-helper-addon-"));
      const selected = path.join(root, "worker.js");
      fs.writeFileSync(selected, "exports.constructions = 0; exports.AsyncWorker = class { " +
    "constructor() { exports.constructions++; } sendToCpp() {} };\n");
      const originalExists = fs.existsSync;
      fs.existsSync = () => {throw new Error("Unexpected fallback addon lookup");};
      try {
        withNativePath(selected, () => {
          const core = helper.create_core.call(quietHelper);
          assert.equal(typeof core.emit_to, "function");
          assert.equal(typeof core.from.on, "function");
          assert.equal(require(selected).constructions, 1);
        });
      } finally {
        fs.existsSync = originalExists;
        delete require.cache[selected];
        fs.rmSync(root, {recursive: true, force: true});
      }
    });

    test("unset and empty compute addon keep default discovery", () => {
      const originalExists = fs.existsSync;
      const expected = path.join(path.dirname(process.execPath), "libs", "mom.node");
      fs.existsSync = (candidate) => {
        assert.equal(candidate, expected);
        // Stop before require(): this fixture must not load a real native addon from the host.
        throw new Error("Default addon discovery reached");
      };
      try {
        for (const selected of [undefined, ""]) {
          withNativePath(selected, () => {
            assert.throws(() => helper.create_core.call(quietHelper),
              /Default addon discovery reached/);
          });
        }
      } finally {
        fs.existsSync = originalExists;
      }
    });
  });

  /** @param {string} input @param {TestEnvironment} [env] */
  function runStdinFixture(input, env = {}) {
    return spawnSync(process.execPath, [stdinFixture], {
      cwd: path.join(__dirname, ".."), input, env: {...process.env, ...env},
      encoding: "utf8", timeout: 5000,
    });
  }

  /** @param {string} [mode] @returns {Promise<{status: number | null, signal: string | null,
 *   stdout: string, stderr: string, error: Error | undefined}>} */
  function runIpcFixture(mode = "") {
    return new Promise((resolve) => {
      const child = fork(shutdownFixture, [], {
        cwd: path.join(__dirname, ".."),
        env: {...process.env, MOM_CLUSTER_WORKER: "1", thread_id: "0",
          ...(mode ? {MOM_HELPER_SHUTDOWN_MODE: mode} : {})},
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        timeout: 5000,
        killSignal: "SIGKILL",
      });
      let stdout = "";
      let stderr = "";
      child.stdout?.setEncoding("utf8").on("data", (chunk) => {stdout += chunk;});
      child.stderr?.setEncoding("utf8").on("data", (chunk) => {stderr += chunk;});
      /** @type {Error | undefined} */
      let error;
      child.on("error", (value) => {error = value;});
      child.on("close", (status, signal) => resolve({status, signal, stdout, stderr, error}));
      child.send({type: "close"});
    });
  }

  test.describe("Worker dispatch, replacement, and shutdown", {concurrency: false}, () => {
    test("subprocess worker failures stay bounded and cleanup remains forceable", () => {
      for (const mode of ["malformed", "oversized", "unwritable", "unexpected-exit"]) {
        /** @type {TestEnvironment} */
        const env = {...process.env, MOM_WORKER_FAILURE_MODE: mode};
        delete env["MOM_CLUSTER_WORKER"];
        delete env["MOM_USE_SUBPROCESS_WORKERS"];
        const result = spawnSync(process.execPath, [fixture], {
          cwd: path.join(__dirname, ".."),
          env,
          encoding: "utf8",
          timeout: 5000,
        });
        assert.equal(result.error, undefined, result.error && result.error.message);
        assert.equal(result.status, 0, `${mode}: ${result.stderr}`);
        assert.equal((result.stdout.match(/HELPER_MALFORMED_WORKER_OK/g) || []).length, 1,
          `${mode}: ${result.stdout}`);
      }
    });

    for (const transport of ["cluster", "subprocess"]) {
      for (const outcome of ["clean", "nonzero", "signal"]) {
        test(`expected close ${transport} ${outcome} exit handling`, () => {
          /** @type {TestEnvironment} */
          const env = {
            ...process.env,
            MOM_WORKER_FAILURE_MODE: `normal-close-${outcome}`,
            MOM_WORKER_TRANSPORT: transport,
          };
          delete env["MOM_CLUSTER_WORKER"];
          delete env["MOM_GPU_BACKEND"];
          if (transport === "cluster") {
            delete env["MOM_USE_SUBPROCESS_WORKERS"];
          } else {
            env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
          }
          const markerDir = outcome === "clean" ?
            fs.mkdtempSync(path.join(os.tmpdir(), "mom-helper-before-exit-")) : null;
          const markerPath = markerDir ? path.join(markerDir, "before-exit") : null;
          if (markerPath) {env["MOM_HELPER_BEFORE_EXIT_MARKER"] = markerPath;}
          const result = spawnSync(process.execPath, [fixture], {
            cwd: path.join(__dirname, ".."), env, encoding: "utf8", timeout: 5000,
          });
          let beforeExitMarker = "";
          if (markerPath && markerDir) {
            if (fs.existsSync(markerPath)) {beforeExitMarker = fs.readFileSync(markerPath, "utf8");}
            fs.rmSync(markerDir, {recursive: true, force: true});
          }
          assert.equal(result.error, undefined, result.error && result.error.message);
          assert.equal(result.status, outcome === "clean" ? 0 : 1, result.stderr);
          const line = result.stdout.split("\n")
            .find((entry) => entry.startsWith("HELPER_EXPECTED_CLOSE_RESULT "));
          assert.ok(line, result.stdout);
          const observed = JSON.parse(line.slice("HELPER_EXPECTED_CLOSE_RESULT ".length));
          assert.equal(observed.transport, transport);
          assert.equal(observed.mode, `normal-close-${outcome}`);
          assert.equal(observed.workerErrors, 0);
          if (outcome === "clean") {
            assert.equal(observed.workerCode, 0, result.stdout);
            assert.equal(observed.workerSignal, null, result.stdout);
            assert.equal(observed.parentExitCode, 0, result.stdout);
            assert.deepEqual(observed.loggedErrors, [], result.stdout);
            assert.equal(beforeExitMarker, "beforeExit\n", result.stdout);
          } else {
            assert.equal(observed.parentExitCode, 1, result.stdout);
            assert.equal(observed.loggedErrors.length, 1, result.stdout);
            assert.match(observed.loggedErrors[0], /Worker 0 exited unexpectedly/);
            if (outcome === "nonzero") {
              assert.equal(observed.workerCode, 7, result.stdout);
              assert.equal(observed.workerSignal, null, result.stdout);
              assert.match(observed.loggedErrors[0], /with code 7/);
            } else {
              assert.ok(observed.workerSignal !== null ||
            (typeof observed.workerCode === "number" && observed.workerCode !== 0), result.stdout);
              if (observed.workerSignal !== null) {
                assert.match(observed.loggedErrors[0],
                  new RegExp("with signal " + observed.workerSignal));
              } else {
                assert.match(observed.loggedErrors[0],
                  new RegExp("with code " + observed.workerCode));
              }
            }
          }
        });
      }
    }

    test("subprocess worker input accepts bounded envelopes and rejects malformed lines", () => {
      const valid = runStdinFixture('{"type":"close"}\n');
      assert.equal(valid.error, undefined, valid.error && valid.error.message);
      assert.equal(valid.status, 0, valid.stderr);

      const inherited = runStdinFixture('{"type":"__proto__"}\n{"type":"close"}\n');
      assert.equal(inherited.error, undefined, inherited.error && inherited.error.message);
      assert.equal(inherited.status, 1, inherited.stderr);
      assert.match(inherited.stderr, /input was malformed/);

      /** @type {Array<[string, string, RegExp]>} */
      const malformedInputs = [
        ["malformed", "{bad\n", /input was malformed/],
        ["trailing", '{"type":"close"', /input was malformed/],
        ["oversized", "x".repeat(1024 * 1024 + 1), /message line limit/],
      ];
      for (const [name, input, expected] of malformedInputs) {
        const result = runStdinFixture(input);
        assert.equal(result.error, undefined, `${name}: ${result.error && result.error.message}`);
        assert.equal(result.status, 1, `${name}: ${result.stderr}`);
        assert.match(result.stderr, expected);
        assert.doesNotMatch(result.stderr, /\{bad|x{64}/);
      }
    });

    test("worker dispatch stamps the final bench job topology", () => {
      const job = {
        type: "bench",
        job: {
          algo: "cn/gpu",
          blob_hex: "00",
          dev: "gpu1*[intensity=640],gpu1*[intensity=640],gpu1*[intensity=640]",
        },
      };
      const result = runStdinFixture(
        JSON.stringify(job) + "\n{\"type\":\"close\"}\n",
        {MOM_TEST_THREAD_ID: "2"},
      );
      assert.equal(result.error, undefined, result.error && result.error.message);
      assert.equal(result.status, 0, result.stderr);
      const marker = result.stdout.split("\n")
        .find((line) => line.startsWith("HELPER_STDIN_BENCH "));
      assert.ok(marker, result.stdout);
      const finalJob = JSON.parse(marker.slice("HELPER_STDIN_BENCH ".length));
      assert.equal(finalJob.dev, "gpu1");
      assert.equal(finalJob.intensity, 640);
      assert.equal(finalJob.thread_id, 2);
      assert.equal(finalJob.thread_num, 3);
    });

    test("replacement workers ignore every stale message", () => {
      const result = spawnSync(process.execPath, [generationFixture], {
        cwd: path.join(__dirname, ".."),
        encoding: "utf8",
        timeout: 5000,
      });

      assert.equal(result.error, undefined, result.error && result.error.message);
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /HELPER_WORKER_GENERATION_OK/);
    });

    test("replacement workers force-kill stale subprocesses before cleanup", () => {
      /** @type {TestEnvironment} */
      const env = {...process.env, MOM_USE_SUBPROCESS_WORKERS: "1"};
      delete env["MOM_CLUSTER_WORKER"];
      const result = spawnSync(process.execPath, [replacementFixture], {
        cwd: path.join(__dirname, ".."),
        env,
        encoding: "utf8",
        timeout: 5000,
      });

      assert.equal(result.error, undefined, result.error && result.error.message);
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /HELPER_WORKER_REPLACEMENT_OK\n$/);
      const pidLine = result.stdout.split("\n")
        .find((line) => line.startsWith("HELPER_WORKER_REPLACEMENT_PIDS "));
      assert.ok(pidLine, result.stdout);
      const pids = JSON.parse(pidLine.slice("HELPER_WORKER_REPLACEMENT_PIDS ".length));
      assert.equal(pids.length, 2, result.stdout);
      assert.equal(new Set(pids).size, pids.length, result.stdout);
      for (const pid of pids) {
        assert.throws(() => process.kill(pid, 0),
          (error) => error && typeof error === "object" && "code" in error && error.code === "ESRCH",
          `worker ${pid} was not reaped`);
      }
    });

    for (const {name, input, mode, expected} of [
      {
        name: "issues one native close and waits for cooperative completion",
        input: '{"type":"close"}\n{"type":"close"}\n',
        expected: {code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0},
      },
      {
        name: "ignores valid commands after IPC close",
        input: '{"type":"close"}\n{"type":"pause"}\n',
        expected: {code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0},
      },
      {
        name: "owns clean EOF after an IPC close",
        input: '{"type":"close"}\n',
        expected: {code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0},
      },
      {
        name: "closes on clean stdin EOF",
        input: "",
        expected: {code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0},
      },
      {
        name: "does not force exit on its first signal after IPC close",
        input: '{"type":"close"}\n',
        mode: "ipc-signal",
        expected: {code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0},
      },
      {
        name: "force exits on its second actual signal",
        input: "",
        mode: "second-signal",
        expected: {code: 0, closeCalls: 1, closeEvents: 0, emittedTypes: ["close"], emergencyExits: 1},
      },
    ]) {
      test(`worker shutdown ${name}`, () => {
        /** @type {TestEnvironment} */
        const env = {...process.env};
        delete env["MOM_HELPER_SHUTDOWN_MODE"];
        if (mode) {env["MOM_HELPER_SHUTDOWN_MODE"] = mode;}
        const result = spawnSync(process.execPath, [shutdownFixture], {
          cwd: path.join(__dirname, ".."), input, env, encoding: "utf8", timeout: 5000,
        });
        assert.equal(result.error, undefined, result.error && result.error.message);
        assert.equal(result.status, 0, result.stderr);
        const line = result.stdout.split("\n")
          .find((entry) => entry.startsWith("HELPER_SHUTDOWN_RESULT "));
        assert.ok(line, result.stdout);
        assert.deepEqual(JSON.parse(line.slice("HELPER_SHUTDOWN_RESULT ".length)), expected);
      });
    }

    test("worker shutdown over IPC reaches natural process exit", async () => {
      const result = await runIpcFixture();
      assert.equal(result.error, undefined, result.error && result.error.message);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.signal, null, result.stderr);
      const line = result.stdout.split("\n")
        .find((entry) => entry.startsWith("HELPER_SHUTDOWN_RESULT "));
      assert.ok(line, result.stdout);
      assert.deepEqual(JSON.parse(line.slice("HELPER_SHUTDOWN_RESULT ".length)), {
        code: 0, closeCalls: 1, closeEvents: 1, emittedTypes: ["close"], emergencyExits: 0,
      });
    });
  });
});
