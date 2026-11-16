"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {describe, test} = require("node:test");
const {spawnSync} = require("node:child_process");
const {parseMemAvailableBytes, verifyNativeCiHostHeadroom, verifyNativeCiMemoryEnvelope} = require("../scripts/native-ci-memory-envelope");
const repo = path.resolve(__dirname, "..");

/** @param {import('node:test').TestContext} t */
function cgroupFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cgroup-")), directory = path.join(root, "system.slice", "run-build.scope");
  fs.mkdirSync(directory, {recursive: true});
  fs.writeFileSync(path.join(directory, "memory.max"), "5368709120\n", {flag: "wx"});
  fs.writeFileSync(path.join(directory, "memory.swap.max"), "0\n", {flag: "wx"});
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, directory};
}


describe("Native CI memory containment", () => {
  test("Linux source checks install their native fixture prerequisites explicitly", () => {
    const workflow = fs.readFileSync(path.join(repo, ".github/workflows/build-release-artifacts.yml"), "utf8"),
      sourceJob = workflow.split("  source-quality:\n")[1]?.split("  windows-script-quality:\n")[0] || "",
      packages = sourceJob.match(/sudo apt-get install[^\n]+/)?.[0].split(/\s+/) || [];
    for (const dependency of ["cmake", "ninja-build", "patchelf", "libboost-dev", "libuv1-dev", "libssl-dev", "strace", "unzip"]) {
      assert.ok(packages.includes(dependency), `${dependency} is required by the offline native fixtures`);
    }
    assert.ok(sourceJob.indexOf("sudo apt-get install") < sourceJob.indexOf("npm test -- host"));
  });

  test("admits only the exact 5 GiB, swap-zero cgroup-v2 envelope", t => {
    const fixture = cgroupFixture(t);
    assert.equal(verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), fixture.directory);
    assert.throws(() => verifyNativeCiMemoryEnvelope("2:memory:/legacy\n", fixture.root), /unified cgroup-v2/);
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/../outside\n", fixture.root), /invalid unified cgroup path/);
    fs.writeFileSync(path.join(fixture.directory, "memory.max"), "max\n");
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), /memory.max envelope/);
  });

  test("fails closed on missing or redirected cgroup controls", t => {
    const fixture = cgroupFixture(t);
    fs.unlinkSync(path.join(fixture.directory, "memory.swap.max"));
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), /memory.swap.max/);
    const outside = path.join(fixture.root, "outside");
    fs.mkdirSync(outside); fs.symlinkSync(outside, path.join(fixture.root, "linked"));
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/linked\n", fixture.root), /escaped the cgroup-v2 mount/);
  });

  test("requires strict host MemAvailable headroom before creating the producer scope", () => {
    const requiredBytes = 5 * 1024 ** 3 + 512 * 1024 ** 2,
      atLimit = `MemTotal: 16384000 kB\nMemAvailable: ${requiredBytes / 1024} kB\n`,
      aboveLimit = `MemTotal: 16384000 kB\nMemAvailable: ${requiredBytes / 1024 + 1} kB\n`;
    assert.equal(parseMemAvailableBytes(aboveLimit), requiredBytes + 1024);
    assert.throws(() => verifyNativeCiHostHeadroom(atLimit), /more than 5 GiB plus 512 MiB/);
    assert.deepEqual(verifyNativeCiHostHeadroom(aboveLimit), {availableBytes: requiredBytes + 1024, requiredBytes});
    assert.throws(() => parseMemAvailableBytes("MemAvailable: 10 kB\nMemAvailable: 11 kB\n"), /one MemAvailable/);
    assert.throws(() => parseMemAvailableBytes("MemAvailable: unlimited kB\n"), /malformed/);
    assert.throws(() => parseMemAvailableBytes("MemAvailable: 9007199254740992 kB\n"), /overflows/);
  });

  test("wrapper verifies before exec and uses a non-root memory-limited systemd scope", () => {
    const wrapper = fs.readFileSync(path.join(repo, "scripts/run-bounded-native-ci.sh"), "utf8");
    if (process.platform === "linux") {
      const syntax = spawnSync("bash", ["-n", path.join(repo, "scripts/run-bounded-native-ci.sh")], {encoding: "utf8"});
      assert.equal(syntax.status, 0, syntax.stderr);
    }
    assert.match(wrapper, /sudo_path=\/usr\/bin\/sudo/);
    assert.match(wrapper, /systemd_run_path=\/usr\/bin\/systemd-run/);
    assert.match(wrapper, /runuser_path=\/usr\/sbin\/runuser/);
    assert.match(wrapper, /env -0 \| "\$sudo_path" -n "\$systemd_run_path"/);
    assert.match(wrapper, /-p MemoryMax=5368709120 -p MemorySwapMax=0 --/);
    assert.doesNotMatch(wrapper, /MemoryOOMGroup|sudo_path" -n -E/);
    assert.match(wrapper, /runuser_path.*--user "\$original_user"/);
    assert.match(wrapper, /read -r -d '' entry[\s\S]*?export "\$entry"/);
    assert.match(wrapper, /native-ci-memory-envelope\.js[\s\S]*?exec "\$@"/);
    assert.match(wrapper, /env -i "\$verifier_node"[\s\S]*?read -r -d '' entry/);
    assert.match(wrapper, /env -i "\$verifier_node" "\$script_dir\/native-ci-memory-envelope\.js" --host-headroom/);
    assert.ok(wrapper.indexOf("--host-headroom") < wrapper.indexOf('"$systemd_run_path"'), "host memory is admitted before scope creation");
    assert.match(wrapper, /\(\(EUID != 0\)\)[\s\S]*?original_uid/);
    assert.doesNotMatch(wrapper, /command -v/);
    assert.doesNotMatch(wrapper, /\beval\b|bash\s+-c|sh\s+-c/);

    const prerequisites = fs.readFileSync(path.join(repo, ".github/workflows/build-native-prerequisites.yml"), "utf8"),
      linuxComponent = fs.readFileSync(path.join(repo, ".github/workflows/build-linux-native-component.yml"), "utf8"),
      linuxLlvm = fs.readFileSync(path.join(repo, ".github/workflows/build-linux-llvm-job.yml"), "utf8"),
      windowsGpu = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-gpu-job.yml"), "utf8"),
      windowsLlvm = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-llvm-job.yml"), "utf8"),
      windowsJit = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-jit-job.yml"), "utf8"),
      windowsDpcpp = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-dpcpp-job.yml"), "utf8"),
      windowsHost = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-host-job.yml"), "utf8"),
      windowsApplication = fs.readFileSync(path.join(repo, ".github/workflows/build-windows-application-job.yml"), "utf8");
    assert.match(prerequisites, /run: scripts\/run-bounded-native-ci\.sh node scripts\/prepare-native-ci\.js stage rocm-comgr-sdk/);
    assert.ok(prerequisites.includes("run: scripts/run-bounded-native-ci.sh node scripts/prepare-native-ci.js stage acpp-cuda-hip-sdk"),
      "The compiler-heavy ACPP SDK stage must enforce the host memory envelope");
    assert.match(prerequisites, /scripts\/run-bounded-native-ci\.sh node scripts\/prepare-native-ci\.js stage linux-native-application/);
    assert.match(windowsGpu, /run: scripts\/run-bounded-native-ci\.sh node scripts\/run-native-windows-gpu-ci\.js "\$MOM_NATIVE_WINDOWS_GPU_PROFILE"/);
    const stageJobs = [["build-linux-cuda-images", "cuda-images-linux"], ["support-linux-source-jit", "linux-jit-support"],
      ["targets-linux-source-jit", "linux-jit-targets"], ["frontend-linux-source-jit", "linux-jit-frontend"],
      ["link-linux-source-jit", "linux-jit-link"], ["build-linux-static-dpcpp-sdk", "linux-jit-static-dpcpp"],
      ["build-llvm-common", "llvm-common"], ["build-llvm-core-codegen", "llvm-core-codegen"],
      ["build-llvm-backends-clang", "llvm-backends-clang"],
      ["build-clang-frontend", "clang-frontend"],
      ["build-linux-rocm-host-sdk", "rocm-host-sdk"], ["build-linux-hermes-sdk", "hermes-sdk"],
      ["build-linux-openmp-sdk", "openmp-sdk"], ["build-linux-io-openssl-sdk", "io-openssl"],
      ["build-linux-zstd-sdk", "zstd-sdk"], ["prepare-linux-jit-resources", "linux-jit-resources"],
      ["configure-linux-sdk", "configure"], ["verify-linux-nvrtc", "nvrtc-linux"]];
    for (const [jobId = "", stage = ""] of stageJobs) {
      const marker = `\n  ${jobId}:`, start = prerequisites.indexOf(marker), remaining = prerequisites.slice(start + marker.length),
        next = /\n\x20{2}[a-z0-9-]+:\n/.exec(remaining), job = start >= 0 ? prerequisites.slice(start,
          next ? start + marker.length + next.index : undefined) : "";
      if (jobId.startsWith("build-llvm-") || jobId.startsWith("build-clang-") || jobId === "build-lld" || jobId === "build-tools") {
        assert.match(job, /uses: \.\/\.github\/workflows\/build-linux-llvm-job\.yml/);
        assert.ok(job.includes(`stage: ${stage}`), stage);
      } else {
        assert.match(job, /timeout-minutes: 60/, jobId + " retains its one-hour job limit");
        assert.ok(job.includes(`scripts/run-bounded-native-ci.sh node scripts/prepare-native-ci.js stage ${stage}`), stage);
      }
    }
    assert.ok(linuxLlvm.includes("timeout-minutes: " + String.fromCharCode(36) +
      "{{ inputs.stage == 'llvm-common' && 30 || inputs.stage == 'llvm-backends-clang' && 30 || 60 }}"),
    "merged common and backend/Clang groups have 30-minute limits; other chunks stay within one hour");
    assert.match(linuxLlvm, /scripts\/run-bounded-native-ci\.sh node scripts\/prepare-native-ci\.js stage/);
    assert.match(prerequisites, /run: scripts\/run-bounded-native-ci\.sh node scripts\/prepare-native-ci\.js stage install-audit/);
    assert.match(linuxComponent, /timeout-minutes: 50\n\s+env:[\s\S]*?run: scripts\/run-bounded-native-ci\.sh node scripts\/prepare-native-ci\.js stage linux-native-component/);
    assert.match(windowsLlvm, /timeout-minutes: 45\n\s+env:[\s\S]*?run: scripts\/run-bounded-native-ci\.sh node scripts\/run-native-windows-llvm-ci\.js/);
    assert.match(windowsJit, /timeout-minutes: 45[\s\S]*?exec scripts\/run-bounded-native-ci\.sh node scripts\/run-native-windows-jit-ci\.js/);
    assert.match(windowsDpcpp, /timeoutMs = remaining\(capMs\)[\s\S]*?memoryScope = path\.join\(repo, "scripts\/run-bounded-native-ci\.sh"\)[\s\S]*?spawnSync\(memoryScope, \[timeout,[\s\S]*?timeout: timeoutMs \+ 10000/);
    assert.match(windowsHost, /timeoutMs = boundedTimeout\(capMs\)[\s\S]*?memoryScope = path\.join\(repo, "scripts\/run-bounded-native-ci\.sh"\)[\s\S]*?spawnSync\(memoryScope, \["\/usr\/bin\/timeout",[\s\S]*?timeout: timeoutMs \+ 7_000/);
    assert.match(windowsApplication, /timeoutMs = remaining\(45 \* 60 \* 1000, 5 \* 60 \* 1000\)[\s\S]*?memoryScope = path\.join\(repo, "scripts\/run-bounded-native-ci\.sh"\)[\s\S]*?spawnSync\(memoryScope, \[timeout,[\s\S]*?npm, "run", "build:release"/);
  });

  test("restores multiline environment and literal argv only in the non-root child", {
    skip: process.platform !== "linux" || process.getuid?.() === 0,
  }, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-env-pipe-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    // This fixture bypasses only the kernel check to isolate shell transport. The verifier's
    // real cgroup admission is covered above and by the separately recorded live-scope smoke.
    const verifier = path.join(root, "verifier-node");
    fs.writeFileSync(verifier, "#!/bin/sh\n[ -z \"$NODE_OPTIONS$NODE_PATH\" ] || exit 23\nexit 0\n", {mode: 0o700, flag: "wx"});
    const env = {PATH: root + ":/usr/bin:/bin", MOM_CI_ENV_FIXTURE: "line1\nline2", CI_LITERAL: "$(not-a-command)"};
    const input = Object.entries(env).map(([key, value]) => key + "=" + value + "\0").join("");
    const result = spawnSync("/bin/bash", [path.join(repo, "scripts/run-bounded-native-ci.sh"),
      "--verify", String(process.getuid?.()), verifier, "--", process.execPath, "-e",
      "console.log(JSON.stringify({value:process.env.MOM_CI_ENV_FIXTURE,literal:process.env.CI_LITERAL,arg:process.argv[1],uid:process.getuid()}))",
      "literal $(not-a-command)"], {input, encoding: "utf8"});
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {value: "line1\nline2", literal: "$(not-a-command)",
      arg: "literal $(not-a-command)", uid: process.getuid?.()});
  });

  test("producer PATH and Node preloads cannot replace the verifier", {
    skip: process.platform !== "linux" || process.getuid?.() === 0,
  }, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-verifier-env-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const verifier = path.join(root, "verifier-node");
    // Isolate verifier dispatch/environment; actual kernel admission is tested separately.
    fs.writeFileSync(verifier, "#!/bin/sh\n[ -z \"$NODE_OPTIONS$NODE_PATH\" ] || exit 23\nprintf 'verified\\n'\n", {mode: 0o700, flag: "wx"});
    fs.writeFileSync(path.join(root, "node"), "#!/bin/sh\nprintf 'BYPASSED\\n'\nexit 0\n", {mode: 0o700, flag: "wx"});
    const result = spawnSync("/bin/bash", [path.join(repo, "scripts/run-bounded-native-ci.sh"),
      "--verify", String(process.getuid?.()), verifier, "--", "/usr/bin/env"], {
      env: {...process.env, NODE_OPTIONS: "--require=/missing-preload", NODE_PATH: root},
      input: "PATH=" + root + ":/usr/bin:/bin\0NODE_OPTIONS=--require=/producer-only\0", encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /^verified\n/);
    assert.match(result.stdout, /NODE_OPTIONS=--require=\/producer-only/);
    assert.doesNotMatch(result.stdout, /BYPASSED/);
  });
});
