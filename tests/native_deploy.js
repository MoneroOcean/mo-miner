"use strict";

const assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {spawn, spawnSync} = require("node:child_process");
const {deploy, deployLinux, windowsDeployOptions} = require("../scripts/test-deploy");

/** @param {NodeJS.ProcessEnv} [env]
 * @param {{ids?: string[], statuses?: number[], missing?: boolean, inventory?: string}} [fixture]
 */
function windows(env = {}, fixture = {}) {
  /** @type {{command: string, args: string[], env: NodeJS.ProcessEnv}[]} */
  const calls = [];
  /** @type {string[]} */
  const messages = [];
  const statuses = [...(fixture.statuses || [])];
  const status = deploy("win32", "x64", env, (command, args, options) => {
    calls.push({command, args, env: {...options.env}});
    if (args.includes("-Command")) {
      return {status: 0, stdout: fixture.inventory ?? JSON.stringify(fixture.ids || ["8086", "10DE", "1002"])};
    }
    return {status: statuses.shift() ?? 0};
  }, () => !fixture.missing, line => messages.push(line));
  return {calls, messages, status};
}

test.describe("Native deployment dispatcher", () => {
  test("host lane accepts Windows summary CRLF and rejects invalid counts safely", {
    skip: process.platform !== "linux",
  }, () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-deploy.sh"), "utf8");
    const start = source.indexOf("run_lane() {");
    const end = source.indexOf("\nquiet_phase() {", start);
    assert.ok(start > 0 && end > start);
    for (const [line, outcome, counts] of [
      ["MOM_TEST_SUMMARY 2 2 0 0\n", "PASS", "2 2 0 0"],
      ["MOM_TEST_SUMMARY 2 2 0 0\r\n", "PASS", "2 2 0 0"],
      ["MOM_TEST_SUMMARY 02 02 00 00\r\r\n", "PASS", "2 2 0 0"],
      ["MOM_TEST_SUMMARY 08 08 00 00\r\n", "PASS", "8 8 0 0"],
      ["MOM_TEST_SUMMARY 1 0 0 1\r\r\n", "SKIP", "1 0 0 1"],
      ["MOM_TEST_SUMMARY 1 0 1 0\r\n", "FAIL", "1 0 1 0"],
      ...["0 0 0 0", "00 00 00 00", "1 0 0 0", "1 1 1 0", "1 1 0 1", "0 1 0 0"]
        .map(value => [`MOM_TEST_SUMMARY ${value}\n`, "FAIL", "0 0 0 0"]),
      ...["-1", "1+1", "name", "99999999999999999999", "0 extra", "0\rX"]
        .map(value => [`MOM_TEST_SUMMARY 1 1 0 ${value}\n`, "FAIL", "0 0 0 0"]),
    ]) {
      const code = `set -euo pipefail; TEST_COUNTS=(0 0 0 0); INTERRUPTED_STATUS=0
        run_active() { printf '%s' "$SUMMARY_FIXTURE" > "$2"; }
        record_result() { printf 'RESULT %s %s\\n' "$2" "$3"; }
        ${source.slice(start, end)}
        run_lane windows-amd fixture
        printf 'COUNTS %s\\n' "\${TEST_COUNTS[*]}"`;
      const result = spawnSync("bash", ["-c", code], {
        encoding: "utf8", timeout: 10000, env: {...process.env, SUMMARY_FIXTURE: line},
      });
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0);
      assert.match(result.stdout, new RegExp(`RESULT ${outcome} `));
      assert.ok(result.stdout.includes(`COUNTS ${counts}\n`));
      assert.doesNotMatch(result.stderr, /arithmetic|syntax error/);
      if (counts === "0 0 0 0") {assert.match(result.stdout, /invalid test summary/);}
    }
  });

  test("host stream shows benchmark starts, deadlines and cleanup without losing raw output or exit", {
    skip: process.platform !== "linux",
  }, () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-deploy.sh"), "utf8");
    const start = source.indexOf("stream_lane() {");
    const end = source.indexOf("\nrun_lane() {", start);
    assert.ok(start > 0 && end > start);
    const lines = ["[perf] zelhash: starting benchmark (deadline 300000 ms)",
      "Release command deadline reached after 300000 ms; waiting for natural completion.",
      "Release benchmark still waiting for cooperative cleanup.",
      "ERROR: Compute core error: PRIVATE_FIXTURE_DETAIL", "[deploy] Debug logs: fixture/debug",
      "PRIVATE_FIXTURE_NOISE"];
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-deploy-stream-fixture-"));
    const log = path.join(root, "raw.log");
    try {
      const code = `set -euo pipefail; ${source.slice(start, end)}
        DEPLOY_SKIP_VECTORS=0
        emit() { printf '%s\\n' ${lines.map(line => JSON.stringify(line)).join(" ")}; return 17; }
        set +e; stream_lane "$1" emit; exit $?`;
      const result = spawnSync("bash", ["-c", code, "fixture", log], {encoding: "utf8", timeout: 10000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 17);
      for (const line of lines.slice(0, -1).filter(line => !line.includes("Compute core error:"))) {
        assert.ok(result.stdout.includes(line));
      }
      assert.match(result.stdout, /\[perf\] Compute core error reported; see private debug logs/);
      assert.equal(result.stdout.includes("PRIVATE_FIXTURE_DETAIL"), false);
      assert.equal(result.stdout.includes("PRIVATE_FIXTURE_NOISE"), false);
      for (const line of lines) {assert.ok(fs.readFileSync(log, "utf8").includes(line));}
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("Windows lane cancellation waits for its runner without force escalation", {
    skip: process.platform !== "linux",
  }, () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-deploy.sh"), "utf8");
    const start = source.indexOf("interrupt() {");
    const end = source.indexOf("\nprepare_windows_auth() {", start);
    assert.ok(start > 0 && end > start);
    for (const [signal, status] of [["INT", 130], ["TERM", 143]]) {
      const code = `exec 3>&2; INTERRUPTED_STATUS=0; ACTIVE_LANE_PGID=4242; ACTIVE_LANE_WINDOWS=1
        polls=0
        kill() {
          if [ "$1" = -TERM ]; then printf 'RELAY TERM %s\\n' "$3"; return 0; fi
          if [ "$1" = -0 ]; then polls=$((polls+1)); [ "$polls" -le 2 ]; return; fi
          printf 'UNEXPECTED_SIGNAL\\n'; return 1
        }
        sleep() { printf 'WAITING_FOR_RUNNER\\n'; }
        ${source.slice(start, end)}
        interrupt ${signal} ${status}; interrupt ${signal} ${status}
        printf 'CALLER_STATUS %s\\n' "$INTERRUPTED_STATUS"`;
      const result = spawnSync("bash", ["-c", code], {encoding: "utf8", timeout: 10000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, 0);
      assert.equal((result.stdout.match(/RELAY TERM -4242/g) || []).length, 1);
      assert.equal((result.stdout.match(/WAITING_FOR_RUNNER/g) || []).length, 2);
      assert.match(result.stdout, new RegExp(`CALLER_STATUS ${status}\\b`));
      assert.doesNotMatch(result.stdout, /KILL|UNEXPECTED_SIGNAL/);
    }
  });

  test("sudo PTY Windows lane relays TERM promptly and retains delayed cleanup", {
    skip: process.platform !== "linux" || process.env["MOM_TEST_SUDO_PTY"] !== "1", timeout: 45000,
  }, async () => {
    assert.equal(spawnSync("sudo", ["-v"], {stdio: "inherit"}).status, 0);
    const source = fs.readFileSync(require.resolve("../scripts/test-deploy.sh"), "utf8");
    const start = source.indexOf("stream_lane() {");
    const end = source.indexOf("\nrun_lane() {", start);
    const block = source.slice(start, end);
    const rootStart = source.indexOf("run_windows_root() {");
    const rootEnd = source.indexOf("\npackage_windows_release() {", rootStart);
    assert.ok(rootStart > 0 && rootEnd > rootStart);
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sudo-stream-fixture-"));
    try {
      for (const label of ["protected"]) {
        const log = path.join(root, `${label}.log`);
        const worker = `cleanup() { trap '' INT TERM HUP PIPE;
          for i in 1 2 3 4 5; do printf '[perf] fixture: cleanup-%s\\n' "$i"; sleep 0.05; done
          printf '[perf] fixture: EXPORTED\\n'; exit 143; }
          term_cleanup() { printf '[perf] fixture: TERM_RECEIVED\\n'; cleanup; }
          trap term_cleanup TERM; printf '[perf] fixture: READY\\n'; sleep 10; cleanup`;
        const quote = /** @param {string} value */ value => `'${value.replace(/'/g, "'\\''")}'`;
        const code = `set -u -o pipefail; ${block}
          ${source.slice(rootStart, rootEnd)}
          DEPLOY_SKIP_VECTORS=0
          SUDO_PASSWORD=''
          test_windows_fixture() { run_windows_root bash -c ${quote(worker)}; }
          export DEPLOY_SKIP_VECTORS SUDO_PASSWORD; export -f stream_lane run_windows_root test_windows_fixture
          set -m; (trap - INT TERM; exec bash -c 'set -u -o pipefail; stream_lane "$1" test_windows_fixture' fixture "$1") & lane=$!
          set +m; printf 'FIXTURE_LANE %s\\n' "$lane"; wait "$lane"; status=$?
          while kill -0 -- "-$lane" 2>/dev/null; do sleep 0.05; done
          exit "$status"`;
        const child = spawn("bash", ["-c", code, "fixture", log], {stdio: ["inherit", "pipe", "pipe"]});
        let output = "";
        let signaled = false;
        let signalTime = 0;
        const result = await new Promise((resolve, reject) => {
          child.once("error", reject);
          child.stdout.on("data", chunk => {
            output += String(chunk);
            const lane = output.match(/FIXTURE_LANE (\d+)/);
            if (!signaled && lane && output.includes("[perf] fixture: READY")) {
              signaled = true;
              signalTime = Date.now();
              process.kill(-Number(lane[1]), "SIGTERM");
            }
          });
          child.stderr.resume();
          child.once("close", resolve);
        });
        assert.equal(signaled, true);
        assert.equal(result, 143);
        assert.ok(Date.now() - signalTime < 5000, "cleanup must not wait for foreground sleep");
        const raw = fs.readFileSync(log, "utf8");
        assert.equal(raw.includes("[perf] fixture: EXPORTED"), true);
        if (label === "protected") {
          assert.ok(raw.includes("[perf] fixture: TERM_RECEIVED"));
          for (let index = 1; index <= 5; index++) {assert.ok(raw.includes(`cleanup-${index}`));}
        }
      }
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("Linux wrapper waits for delayed child teardown after SIGINT and SIGTERM", {skip: process.platform !== "linux", timeout: 10000}, async () => {
    const modulePath = require.resolve("../scripts/test-deploy");
    for (const [signal, status] of /** @type {[NodeJS.Signals, number][]} */ ([["SIGINT", 130], ["SIGTERM", 143]])) {
      const worker = `process.on(${JSON.stringify(signal)}, () => setTimeout(() => { console.log("CLEANUP_DONE"); process.exit(${status}); }, 150)); console.log("READY"); setInterval(() => {}, 1000);`;
      const code = `const {spawn}=require("node:child_process");
      require(${JSON.stringify(modulePath)}).deployLinux(process.env, () =>
        spawn(process.execPath, ["-e", ${JSON.stringify(worker)}], {stdio:"inherit"}))
        .then(status => {process.exitCode=status;});`;
      const child = spawn(process.execPath, ["-e", code], {stdio: ["ignore", "pipe", "pipe"]});
      let output = "";
      let signaled = false;
      const result = await new Promise((resolve, reject) => {
        child.once("error", reject);
        child.stdout.on("data", chunk => {
          output += String(chunk);
          if (!signaled && output.includes("READY")) {
            signaled = true;
            child.kill(signal);
          }
        });
        child.once("close", resolve);
      });
      assert.equal(result, status);
      assert.match(output, /CLEANUP_DONE/);
    }
  });

  test("Linux delegates unchanged shell entrypoint without changing environment", async () => {
    const env = {MOM_DEPLOY_TARGET: "linux-amd"};
    let count = 0;
    assert.equal(await deployLinux(env, (command, args, options) => {
      count++;
      assert.equal(command, "bash");
      assert.match(args[0] || "", /scripts[/\\]test-deploy\.sh$/);
      assert.equal(options.env, env);
      return spawn(process.execPath, ["-e", "process.exitCode = 23"], {stdio: "inherit"});
    }), 23);
    assert.equal(count, 1);
  });

  test("Windows builds once and runs existing gates sequentially without shell, VM or Docker", () => {
    const result = windows({MOM_RELEASE_VERSION: "v9.2.1"});
    assert.equal(result.status, 0);
    assert.equal(result.calls.length, 6);
    assert.match(result.calls[0]?.args[0] || "", /scripts[/\\]build-release\.js$/);
    assert.deepEqual(result.calls.slice(2, 5).map(call => call.args.at(-1)),
      ["intel-windows", "nvidia-windows", "amd-windows"]);
    assert.equal(result.calls[5]?.args.at(-1), "gpu-multi");
    for (const call of result.calls.slice(2)) {
      assert.equal(call.command, "powershell.exe");
      assert.ok(call.args.includes("-NoProfile"));
      assert.ok(call.args.includes("-ExecutionPolicy"));
      assert.match(call.args[call.args.indexOf("-Archive") + 1] || "", /mom-v9\.2\.1-win\.zip$/);
      assert.equal(call.args[call.args.indexOf("-Archive") + 1], "mom-v9.2.1-win.zip");
    }
    assert.equal(result.calls.some(call => /bash|docker|run\.sh/.test(call.command)), false);
    assert.equal(result.calls[5]?.env["MOM_GPU_TEST_VENDORS"], "intel,nvidia,amd");
    for (const call of result.calls.slice(2, 5)) {assert.ok(call.args.includes("-SkipAbsentGpu"));}
  });

  test("targets retain vendor selection and reject Linux/unknown targets before spawning", () => {
    for (const vendor of ["intel", "nvidia", "amd"]) {
      const result = windows({MOM_DEPLOY_TARGET: `windows-${vendor}`});
      assert.equal(result.calls.length, 3);
      assert.equal(result.calls[2]?.args.at(-1), `${vendor}-windows`);
    }
    for (const target of ["linux", "linux-nvidia", "nvidia", "multi-windows", "unknown"]) {
      assert.throws(() => windows({MOM_DEPLOY_TARGET: target}), /MOM_DEPLOY_TARGET/);
    }
    assert.throws(() => deploy("darwin", "x64", {}), /supports Linux and Windows/);
    assert.throws(() => deploy("win32", "arm64", {}), /supports Linux and Windows/);
  });

  test("Intel release checks leave discrete-device selection to MoM rather than pinning the iGPU", () => {
    const script = fs.readFileSync(require.resolve("../scripts/test-windows-release-gpu.ps1"), "utf8");
    assert.match(script, /\$env:ONEAPI_DEVICE_SELECTOR = "level_zero:gpu"/);
    assert.doesNotMatch(script, /\$env:ZE_AFFINITY_MASK\s*=/);
  });

  test("Windows deployment retains only opt-in evidence outside extraction cleanup", () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-windows-release-gpu.ps1"), "utf8");
    const wrapper = fs.readFileSync(require.resolve("../scripts/test-deploy.sh"), "utf8");
    assert.match(source, /build\\deploy-win\\debug\\\{0\}-\{1\}/);
    assert.match(source, /\$vectorLogPath\s*=\s*Join-Path\s+\$debugDir\s+["']vectors\.log["']/);
    assert.match(source, /\$env:MOM_RELEASE_PERF_EVIDENCE_DIR\s*=/);
    assert.match(source, /\$deployDebug = \$env:MOM_DEPLOY_DEBUG -eq "1"/);
    assert.match(source, /\$debugDir = if \(\$deployDebug\)/);
    assert.match(source, /else \{ Join-Path \$work "diagnostics" \}/);
    assert.match(source, /if \(\$deployDebug -and -not \$env:MOM_RELEASE_PERF_EVIDENCE_DIR\)/);
    assert.match(source, /if \(\$deployDebug\) \{\s*Write-Host "\[deploy\] Debug logs: \$debugDir"/);
    assert.doesNotMatch(source, /MOM_ZELHASH_TIMING|temporaryObserver|zel-trace|winsat-fixtures|validate-cpu-capture/);
    assert.match(wrapper, /MOM_DEPLOY_DEBUG="\$\{MOM_DEPLOY_DEBUG:-\}"/);
    assert.match(wrapper, /if \[ "\$\{MOM_DEPLOY_DEBUG:-\}" = 1 \]; then\s*printf '\[deploy\] Debug logs:/);
    assert.doesNotMatch(source, /Remove-Item[^\r\n]*\$(?:vectorLogPath|debugDir|performanceDir|installLogPath)\b/);
    assert.match(source, /Remove-Item -LiteralPath \$work -Recurse -Force/);
  });

  test("real PowerShell cleans normal logs and retains opt-in evidence and caller environment", {
    skip: process.platform !== "win32", timeout: 60000,
  }, () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-windows-release-gpu.ps1"), "utf8");
    const expand = "Expand-Archive -Force -Path $archivePath -DestinationPath $work";
    const node = '$node = Join-Path $release "mom-node.exe"';
    assert.equal(source.split(expand).length, 2);
    assert.equal(source.split(node).length, 2);
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-debug-fixture-"));
    const temp = path.join(root, "temp");
    const release = path.join(root, "fixture-release");
    const scripts = path.join(root, "scripts");
    const vectors = path.join(root, ".github", "workflows", "scripts");
    try {
      for (const directory of [temp, release, scripts, vectors]) {fs.mkdirSync(directory, {recursive: true});}
      fs.writeFileSync(path.join(root, "mom-v9.2.1-win.zip"), "fixture archive; expansion substituted");
      fs.writeFileSync(path.join(release, "install.bat"),
        "@echo off\r\necho FIXTURE_INSTALL\r\necho FIXTURE_INSTALL_STDERR 1>&2\r\nexit /b %MOM_FIXTURE_INSTALL_EXIT%\r\n");
      fs.writeFileSync(path.join(root, "README.md"), "fixture readme");
      fs.writeFileSync(path.join(vectors, "test-release-windows.ps1"),
        "param([string]$Archive,[string]$Suite)\nWrite-Output 'FIXTURE_VECTOR'\nexit ([int]$env:MOM_FIXTURE_VECTOR_EXIT)\n");
      fs.writeFileSync(path.join(scripts, "check-release-performance.js"), `
        const fs=require('node:fs'),path=require('node:path');
        const root=process.env.MOM_RELEASE_PERF_EVIDENCE_DIR;
        console.log('FIXTURE_EVIDENCE_ENABLED '+Number(Boolean(root)));
        if(root) {fs.mkdirSync(root,{recursive:true});
        fs.writeFileSync(path.join(root,'zelhash.metadata.json'),JSON.stringify({
          schema:'release-performance-benchmark-v3',exitCode:Number(process.env.MOM_FIXTURE_GATE_EXIT),
          stdout:'FIXTURE_GATE',stderr:'FIXTURE_GATE_STDERR'}));}
        console.log('FIXTURE_GATE');console.error('FIXTURE_GATE_STDERR');
        process.exitCode=Number(process.env.MOM_FIXTURE_GATE_EXIT);
      `);
      const fixture = source.replace(expand,
        'Copy-Item -LiteralPath (Join-Path $root "fixture-release") -Destination (Join-Path $work "mom-v9.2.1") -Recurse')
        .replace(node, `$node = '${process.execPath.replace(/'/g, "''")}'`);
      fs.writeFileSync(path.join(root, "helper.ps1"), fixture);
      const debugRoot = path.join(root, "build", "deploy-win", "debug");
      /** @param {number} installExit @param {number} vectorExit @param {number} gateExit
       * @param {string | undefined} inherited @param {boolean} [debug] */
      const run = (installExit, vectorExit, gateExit, inherited, debug = false) => {
        const before = fs.existsSync(debugRoot) ? fs.readdirSync(debugRoot) : [];
        /** @type {NodeJS.ProcessEnv} */
        const env = {...process.env, TEMP: temp, MOM_FIXTURE_INSTALL_EXIT: String(installExit),
          MOM_FIXTURE_VECTOR_EXIT: String(vectorExit), MOM_FIXTURE_GATE_EXIT: String(gateExit)};
        delete env["MOM_DEPLOY_SKIP_VECTORS"];
        delete env["MOM_DEPLOY_ALGO"];
        if (debug) {env["MOM_DEPLOY_DEBUG"] = "1";} else {delete env["MOM_DEPLOY_DEBUG"];}
        if (inherited === undefined) {delete env["MOM_RELEASE_PERF_EVIDENCE_DIR"];} else {
          env["MOM_RELEASE_PERF_EVIDENCE_DIR"] = inherited;
        }
        const code = `$original=[Environment]::GetEnvironmentVariable('MOM_RELEASE_PERF_EVIDENCE_DIR');
          $failed=$false; try { & './helper.ps1' -Archive 'mom-v9.2.1-win.zip' -Platform 'amd-windows' }
          catch { $failed=$true; Write-Output $_.Exception.Message }
          if ([Environment]::GetEnvironmentVariable('MOM_RELEASE_PERF_EVIDENCE_DIR') -cne $original) { exit 99 }
          Write-Output 'FIXTURE_ENV_RESTORED'; if ($failed) { exit 1 } else { exit 0 }`;
        const result = spawnSync("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
          {cwd: root, env, encoding: "utf8", timeout: 15000});
        assert.equal(result.error, undefined);
        assert.equal(result.status, installExit || vectorExit || gateExit ? 1 : 0);
        assert.match(result.stdout, /FIXTURE_ENV_RESTORED/);
        assert.equal(result.stdout.includes("[deploy] Debug logs: "), debug);
        if (installExit || vectorExit || gateExit) {
          assert.match(result.stdout, new RegExp(`exit code ${installExit || vectorExit || gateExit}\\b`));
        }
        if (!installExit && !vectorExit) {
          assert.match(result.stdout, new RegExp(`FIXTURE_EVIDENCE_ENABLED ${Number(debug || inherited !== undefined)}\\b`));
        }
        const added = (fs.existsSync(debugRoot) ? fs.readdirSync(debugRoot) : [])
          .filter(name => !before.includes(name));
        assert.equal(added.length, debug ? 1 : 0);
        assert.deepEqual(fs.readdirSync(temp), []);
        if (!debug) {
          assert.doesNotMatch(result.stdout, /\[perf\] .*starting benchmark|\[deploy\] Debug logs:/);
        }
        if (debug) {
          assert.match(added[0] || "", /^amd-windows-[a-f0-9]{32}$/);
          const evidence = path.join(debugRoot, added[0] || "");
          const context = JSON.parse(fs.readFileSync(path.join(evidence, "context.json"), "utf8").replace(/^\uFEFF/, ""));
          assert.equal(String(context.helperSha256).toLowerCase(), createHash("sha256").update(fixture).digest("hex"));
          assert.match(fs.readFileSync(path.join(evidence, "install.log"), "utf8"), /FIXTURE_INSTALL/);
          assert.match(fs.readFileSync(path.join(evidence, "install.log"), "utf8"), /FIXTURE_INSTALL_STDERR/);
          if (!installExit) {
            const vectorBytes = fs.readFileSync(path.join(evidence, "vectors.log"));
            const vectorText = vectorBytes[0] === 0xff && vectorBytes[1] === 0xfe
              ? vectorBytes.toString("utf16le") : vectorBytes.toString("utf8");
            assert.match(vectorText, /FIXTURE_VECTOR/);
          }
          if (!installExit && !vectorExit) {
            const gateBytes = fs.readFileSync(path.join(evidence, "performance.log"));
            const gateText = gateBytes[0] === 0xff && gateBytes[1] === 0xfe
              ? gateBytes.toString("utf16le") : gateBytes.toString("utf8");
            assert.match(gateText, /FIXTURE_GATE/);
            assert.match(gateText, /FIXTURE_GATE_STDERR/);
            assert.match(gateText, /(?:^|\r?\n)FIXTURE_GATE_STDERR(?:\r?\n|$)/);
            assert.doesNotMatch(gateText, /NativeCommandError|(?:^|\r?\n)At .*\.ps1/);
            if (inherited !== undefined) {assert.equal(fs.existsSync(path.join(evidence, "performance")), false);}
          }
        }
        if (!installExit && !vectorExit && (debug || inherited !== undefined)) {
          const metadataRoot = inherited === undefined
            ? path.join(debugRoot, added[0] || "", "performance") : path.resolve(root, inherited);
          const metadata = JSON.parse(fs.readFileSync(path.join(metadataRoot, "zelhash.metadata.json"), "utf8"));
          assert.equal(metadata.schema, "release-performance-benchmark-v3");
          assert.equal(metadata.exitCode, gateExit);
          assert.equal(metadata.stdout, "FIXTURE_GATE");
          assert.equal(metadata.stderr, "FIXTURE_GATE_STDERR");
        }
      };
      run(23, 0, 0, "caller-evidence");
      run(23, 0, 0, undefined);
      run(0, 29, 0, undefined);
      run(0, 0, 37, "caller-evidence-failure");
      run(0, 0, 37, undefined);
      run(0, 0, 0, undefined);
      run(0, 0, 0, "caller-evidence-success");
      run(0, 0, 0, undefined, true);
      run(23, 0, 0, undefined, true);
      run(0, 29, 0, undefined, true);
      run(0, 0, 37, undefined, true);
      run(0, 0, 0, "caller-debug-evidence", true);
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("absent devices are skips, not passes; mixed lane requires two present vendors", () => {
    const result = windows({}, {ids: ["10DE"]});
    assert.equal(result.calls.length, 3);
    assert.deepEqual(result.messages.slice(0, 3), ["SKIP intel-windows: no PCI GPU detected",
      "PASS nvidia-windows", "SKIP amd-windows: no PCI GPU detected"]);
    assert.match(result.messages[3] || "", /^SKIP multi-windows:/);
    const absent = windows({}, {ids: []});
    assert.equal(absent.messages.filter(line => line.startsWith("PASS")).length, 0);
    assert.equal(absent.messages.filter(line => line.startsWith("SKIP")).length, 4);
  });

  test("lane failure propagates while independent vendors and mixed gate still run", () => {
    const result = windows({}, {statuses: [0, 27, 0, 13, 0]});
    assert.equal(result.status, 27);
    assert.equal(result.calls.length, 6);
    assert.deepEqual(result.messages, ["FAIL intel-windows", "PASS nvidia-windows",
      "FAIL amd-windows", "PASS multi-windows"]);
  });

  test("native iGPU-only vendors skip and cannot qualify the mixed discrete gate", () => {
    const single = windows({}, {ids: ["8086", "10DE"], statuses: [0, 3, 0]});
    assert.equal(single.status, 0);
    assert.equal(single.calls.length, 4);
    assert.match(single.messages[0] || "", /^SKIP intel-windows: no usable discrete GPU discovered$/);
    assert.match(single.messages.at(-1) || "", /^SKIP multi-windows:/);
    const mixed = windows({}, {statuses: [0, 3, 0, 0, 0]});
    assert.equal(mixed.calls.at(-1)?.env["MOM_GPU_BACKEND"], "nvidia");
    assert.equal(mixed.calls.at(-1)?.env["MOM_GPU_TEST_VENDORS"], "nvidia,amd");
    // Other producers do not share the vendor helper's reserved skip status.
    assert.equal(windows({}, {statuses: [3]}).status, 3);
    assert.equal(windows({}, {statuses: [0, 0, 0, 0, 3]}).status, 3);
  });

  test("opt-in packaged admission validates real PowerShell output without loading addons", {skip: process.platform !== "win32"}, () => {
    const source = fs.readFileSync(require.resolve("../scripts/test-windows-release-gpu.ps1"), "utf8");
    const start = source.indexOf("  if ($SkipAbsentGpu) {");
    const end = source.indexOf("  if ($env:MOM_DEPLOY_SKIP_VECTORS", start);
    assert.ok(start > 0 && end > start);
    const invoke = '& (Join-Path $release "mom.cmd") algorithms 2>&1';
    const block = source.slice(start, end);
    assert.ok(block.includes(invoke));
    /** @param {string[]} lines @param {number} nativeStatus @param {number} expected @param {boolean} [optIn] */
    const check = (lines, nativeStatus, expected, optIn = true) => {
      const emit = lines.map(line => `Write-Output '${line.replace(/'/g, "''")}'`).join("; ");
      const code = `$ErrorActionPreference='Stop'; $SkipAbsentGpu=$${optIn ? "true" : "false"};
        $Platform='intel-windows'; function fake-miner { ${emit}; $global:LASTEXITCODE=${nativeStatus} };
        ${block.replace(invoke, "& fake-miner algorithms 2>&1")}
        Write-Host 'ADMITTED'; exit 0`;
      const result = spawnSync("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
        {encoding: "utf8", timeout: 10000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, expected);
      if (expected === 3) {assert.match(result.stdout, /no usable discrete GPU discovered/);}
    };
    for (const cpu of ["cpu", "cpu*8", "cpu^4", "cpu0*1", "cpu1*1:auto[sycl-opencl]"]) {
      check([`MOM_ALGORITHMS ${JSON.stringify({test: cpu})}`], 0, 3);
    }
    check(["gpu1: Intel fixture", "MOM_ALGORITHMS {\"test\":\"gpu1*1:auto[sycl]\"}"], 0, 0);
    for (const json of ["null", "[]", "[{\"test\":\"cpu0*1\"}]", "{}", "123", "\"text\"", "{\"test\":null}",
      "{\"test\":1}", "{\"test\":{}}", "{\"test\":\"\"}", "{\"test\":\"gpu1*0\"}", "invalid"]) {
      check([`MOM_ALGORITHMS ${json}`], 0, 1);
    }
    check([], 0, 1);
    check(["MOM_ALGORITHMS {\"test\":\"cpu0*1\"}"], 3, 1);
    check(["MOM_ALGORITHMS {\"test\":\"gpu1*1\"}"], 0, 1);
    check(["gpu1: Intel fixture", "MOM_ALGORITHMS {\"test\":\"cpu*8\"}"], 0, 1);
    check(["MOM_ALGORITHMS {\"test\":\"cpu0*1\"}", "MOM_ALGORITHMS {\"test\":\"cpu0*1\"}"], 0, 1);
    check([], 3, 0, false);
  });

  test("build failure prevents inventory and dependent lanes", () => {
    const result = windows({}, {statuses: [31]});
    assert.equal(result.status, 31);
    assert.equal(result.calls.length, 1);
    assert.match(result.messages[0] || "", /GPU lanes not run/);
  });

  test("archive reuse bypasses building only when archive exists", () => {
    const reused = windows({MOM_DEPLOY_REUSE_ARCHIVE: "1"});
    assert.equal(reused.calls.length, 5);
    assert.ok(reused.calls[0]?.args.includes("-Command"));
    let checks = 0;
    let builds = 0;
    assert.equal(deploy("win32", "x64", {MOM_DEPLOY_REUSE_ARCHIVE: "1"}, (_command, args) => {
      if (args.includes("-Command")) {return {status: 0, stdout: "[]"};}
      builds++;
      return {status: 0};
    }, () => ++checks > 1, () => { /* Suppress fixture reports. */ }), 0);
    assert.equal(builds, 1);
    assert.throws(() => windows({}, {missing: true}), /archive missing/);
  });

  test("flags are validated; rebuild, algorithm and skip flags reach existing producers", () => {
    for (const name of ["MOM_DEPLOY_SKIP_VECTORS", "MOM_DEPLOY_REUSE_ARCHIVE", "MOM_FORCE_REBUILD"]) {
      assert.throws(() => windowsDeployOptions({[name]: "yes"}), /must be 0 or 1/);
    }
    assert.throws(() => windowsDeployOptions({MOM_DEPLOY_REUSE_ARCHIVE: "1", MOM_FORCE_REBUILD: "1"}), /cannot be combined/);
    for (const version of ["../bad", "vbad", "1;exit", "1\n"]) {
      assert.throws(() => windowsDeployOptions({MOM_RELEASE_VERSION: version}), /Invalid release version/);
    }
    const result = windows({MOM_FORCE_REBUILD: "1", MOM_DEPLOY_ALGO: "cn/gpu", MOM_DEPLOY_SKIP_VECTORS: "1"});
    assert.equal(result.calls.length, 5);
    assert.equal(result.calls[0]?.env["MOM_FORCE_REBUILD"], "1");
    for (const call of result.calls.slice(2)) {
      assert.equal(call.env["MOM_DEPLOY_ALGO"], "cn/gpu");
      assert.equal(call.env["MOM_DEPLOY_SKIP_VECTORS"], "1");
    }
    assert.match(result.messages.at(-1) || "", /^SKIP multi-windows: vector gates disabled$/);
    assert.match(windows({MOM_DEPLOY_ALGO: "cn/gpu"}).messages.at(-1) || "", /single-algorithm/);
  });

  test("worker environment bindings cannot leak into separate gate processes", () => {
    const result = windows({MOM_GPU_BACKEND: "amd", MOM_NATIVE_PATH: "old.node", MOM_RUNTIME_DIR: "old",
      MOM_GPU_TEST_ALGO: "old", MOM_GPU_TEST_VENDORS: "amd", MOM_REQUIRE_MULTI_GPU_TESTS: "1",
      MOM_RELEASE_ARCHIVE_VALIDATION_ONLY: "1",
      ONEAPI_DEVICE_SELECTOR: "old", ZE_AFFINITY_MASK: "old", ACPP_VISIBILITY_MASK: "old",
      UR_ADAPTERS_FORCE_LOAD: "old", OCL_ICD_FILENAMES: "old"});
    for (const call of result.calls.slice(2, 5)) {
      assert.equal(call.env["MOM_GPU_BACKEND"], undefined);
      assert.equal(call.env["MOM_NATIVE_PATH"], undefined);
      assert.equal(call.env["MOM_REQUIRE_MULTI_GPU_TESTS"], undefined);
      assert.equal(call.env["ONEAPI_DEVICE_SELECTOR"], undefined);
      assert.equal(call.env["MOM_RELEASE_ARCHIVE_VALIDATION_ONLY"], undefined);
      assert.equal(call.env["MOM_SKIP_MSR"], "1");
    }
  });

  test("inventory handles Windows BOM and rejects malformed inventory", () => {
    assert.equal(windows({}, {inventory: "\uFEFF[\"10DE\"]"}).calls.length, 3);
    for (const inventory of ["\"10DE\"", "[\"9999\"]", "{}"]) {
      assert.throws(() => windows({}, {inventory}), /Invalid Windows GPU inventory/);
    }
    assert.throws(() => windows({}, {inventory: "invalid"}), SyntaxError);
  });
});
