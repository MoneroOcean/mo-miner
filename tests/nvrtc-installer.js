"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {createHash} = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {describe, test, after} = require("node:test");

/** @param {string | Buffer} bytes */
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
function setup() {
  if (process.platform !== "linux") {return {work: "", bin: "", archive: "", helper: "", headerSize: 0};}
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-installer-"));
  const prefix = "cuda_nvrtc-linux-x86_64-13.3.33-archive";
  const pins = [
    {file: "lib/libnvrtc.so.13.3.33", size: 120072328,
      sha: "afff6fa2eae856b90f66c2d83bbce4a4177404c506291e3a83486244d86c2815", body: "compiler fixture"},
    {file: "lib/libnvrtc-builtins.so.13.3.33", size: 6555592,
      sha: "7394c640e5761d13d2bbcdbc4b4c5dbac7cb53cd5bc732d78f8a5cb38638e913", body: "builtins fixture"},
    {file: "LICENSE", size: 68070,
      sha: "088381bc2d891e719a2a9398645b00bb45f3b24231473a8283ac7e3e66b8a028", body: "license fixture"},
    {file: "include/nvrtc.h", size: 56371,
      sha: "8d0373dc86a2c68528ddd13247204c82df670636383a0fe7f31b791195181f9d", body: "header fixture"},
  ];
  let source = fs.readFileSync(path.join(__dirname, "../scripts/install-nvrtc-current.sh"), "utf8");
  for (const pin of pins) {
    const dest = path.join(work, prefix, pin.file);
    fs.mkdirSync(path.dirname(dest), {recursive: true}); fs.writeFileSync(dest, pin.body);
    assert.equal(source.split(String(pin.size)).length, 2);
    assert.equal(source.split(pin.sha).length, 2);
    source = source.replace(String(pin.size), String(Buffer.byteLength(pin.body))).replace(pin.sha, hash(pin.body));
  }
  const archive = path.join(work, "archive.tar.xz");
  const packed = spawnSync("tar", ["-cJf", archive, "-C", work, prefix], {encoding: "utf8", timeout: 10000});
  assert.equal(packed.status, 0, packed.stderr);
  // Change pins only in this isolated test copy. Production has no hash override.
  source = source.replace("9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a",
    hash(fs.readFileSync(archive)));
  const helper = path.join(work, "install.sh"); fs.writeFileSync(helper, source);
  const bin = path.join(work, "bin");
  fs.mkdirSync(bin);
  return {work, bin, archive, helper, headerSize: Buffer.byteLength("header fixture")};
}
const {work, bin, archive, helper, headerSize} = setup();
if (work) {
  fs.writeFileSync(path.join(bin, "curl"), `#!/usr/bin/env bash
set -euo pipefail
printf 'download\\n' >> "$NV_TEST_DOWNLOADS"
if [[ \${NV_TEST_FAIL:-} == 1 ]]; then exit 7; fi
while (($#)); do
  if [[ $1 == -o ]]; then shift; output=$1; fi
  shift
done
if [[ \${NV_TEST_BAD_ARCHIVE:-} == 1 ]]; then printf bad > "$output"; else cp -- "$NV_TEST_ARCHIVE" "$output"; fi
if [[ \${NV_TEST_RACE:-} == 1 ]]; then mkdir -- "$NV_TEST_DEST"; printf keep > "$NV_TEST_DEST/foreign"; fi
`, {mode: 0o755});
  fs.writeFileSync(path.join(bin, "mv"), `#!/usr/bin/env bash
set -euo pipefail
for destination; do :; done
case "\${NV_TEST_MOVE_STATUS:-}" in
  valid) cp -a -- "\${@: -2:1}" "$destination"; exit 1;;
  bad) mkdir -- "$destination"; printf keep > "$destination/foreign"; exit 1;;
esac
exec /bin/mv "$@"
`, {mode: 0o755});
}

function fixture() {
  const root = fs.mkdtempSync(path.join(work, "case-"));
  const dest = path.join(root, "managed");
  const legacy = path.join(root, "cuda-old");
  fs.mkdirSync(legacy);
  fs.writeFileSync(path.join(legacy, "libnvrtc.so.12"), "another application owns this SDK");
  return {root, dest, legacy, log: path.join(root, "downloads")};
}
/** @param {ReturnType<typeof fixture>} f @param {string} [option] @param {boolean} [conditional] */
function args(f, option = "", conditional = false) {
  return ["-c", `source "$1"; ${conditional ? 'if install_nvrtc_current "$2" "$3"; then exit 0; else exit 1; fi' : 'install_nvrtc_current "$2" "$3"'}`,
    "test", helper, f.dest, option];
}
/** @param {ReturnType<typeof fixture>} f @param {NodeJS.ProcessEnv} [extra] */
function env(f, extra = {}) {
  return {...process.env, PATH: `${bin}:${process.env["PATH"]}`, NV_TEST_ARCHIVE: archive,
    NV_TEST_DOWNLOADS: f.log, NV_TEST_DEST: f.dest, ...extra};
}
/** @param {ReturnType<typeof fixture>} f @param {string} [option]
 * @param {NodeJS.ProcessEnv} [extra] @param {boolean} [conditional] */
function run(f, option = "", extra = {}, conditional = false) {
  const result = spawnSync("bash", args(f, option, conditional), {
    env: env(f, extra), encoding: "utf8", timeout: 60000, maxBuffer: 1024 * 1024
  });
  assert.ifError(result.error);
  assert.equal(fs.readFileSync(path.join(f.legacy, "libnvrtc.so.12"), "utf8"),
    "another application owns this SDK");
  return result;
}
/** @param {{status: number | null, stderr: string}} result */
function success(result) { assert.equal(result.status, 0, result.stderr); }
/** @param {ReturnType<typeof run>} result */
function failure(result) { assert.notEqual(result.status, 0, result.stdout); }
/** @param {ReturnType<typeof fixture>} f */
function noTemporaryFiles(f) {
  assert.ok(!fs.readdirSync(f.root).some(name => name.startsWith(".mom-nvrtc-")));
}
function installed() { const f = fixture(); success(run(f)); return f; }

describe("Shared NVIDIA compiler provisioning", {skip: process.platform !== "linux"}, () => {
  test("fresh install needs no older SDK and publishes one complete compiler", () => {
    const f = installed();
    success(run(f, "--validate-only"));
    assert.equal(fs.readlinkSync(path.join(f.dest, "lib/libnvrtc.so")), "libnvrtc.so.13.3.33");
    assert.equal(fs.statSync(path.join(f.dest, "include/nvrtc.h")).size, headerSize);
    assert.deepEqual(fs.readdirSync(path.join(f.dest, "lib")).sort(), [
      "libnvrtc-builtins.so.13.3", "libnvrtc-builtins.so.13.3.33",
      "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.13.3.33"
    ]);
    noTemporaryFiles(f);
  });
  test("validation of an absent destination is read-only", () => {
    const f = fixture(); failure(run(f, "--validate-only"));
    assert.equal(fs.existsSync(f.dest), false);
    assert.equal(fs.existsSync(f.log), false);
  });
  test("repeat install neither downloads nor replaces a verified payload", () => {
    const f = installed();
    const file = path.join(f.dest, "lib/libnvrtc.so.13.3.33");
    const before = fs.statSync(file);
    success(run(f));
    assert.equal(fs.statSync(file).ino, before.ino);
    assert.equal(fs.readFileSync(f.log, "utf8"), "download\n");
  });
  test("corrupt runtime is rejected without repair or replacement", () => {
    const f = installed();
    const file = path.join(f.dest, "lib/libnvrtc.so.13.3.33");
    fs.writeFileSync(file, "keep invalid file");
    failure(run(f));
    assert.equal(fs.readFileSync(file, "utf8"), "keep invalid file");
  });
  test("wrong header checksum is rejected even when its size matches", () => {
    const f = installed();
    const file = path.join(f.dest, "include/nvrtc.h");
    fs.writeFileSync(file, Buffer.alloc(headerSize));
    failure(run(f, "--validate-only"));
  });
  test("an existing foreign destination is preserved", () => {
    const f = fixture(); fs.mkdirSync(f.dest);
    fs.writeFileSync(path.join(f.dest, "foreign"), "keep");
    failure(run(f));
    assert.equal(fs.readFileSync(path.join(f.dest, "foreign"), "utf8"), "keep");
    assert.equal(fs.existsSync(f.log), false);
  });
  test("destination symlinks are never followed", () => {
    const f = fixture(); fs.symlinkSync(f.legacy, f.dest);
    failure(run(f)); assert.equal(fs.readlinkSync(f.dest), f.legacy);
  });
  test("payload-directory symlinks are rejected", () => {
    const f = installed();
    fs.renameSync(path.join(f.dest, "lib"), path.join(f.dest, "oldlib"));
    fs.symlinkSync("oldlib", path.join(f.dest, "lib"));
    failure(run(f));
  });
  test("compiler file symlinks are rejected", () => {
    const f = installed();
    const file = path.join(f.dest, "lib/libnvrtc.so.13.3.33");
    fs.renameSync(file, `${file}.original`); fs.symlinkSync(`${file}.original`, file);
    failure(run(f));
  });
  test("unexpected aliases cannot redirect compilation", () => {
    const f = installed(); const file = path.join(f.dest, "lib/libnvrtc.so.13");
    fs.unlinkSync(file); fs.symlinkSync("/foreign/libnvrtc.so.13", file);
    failure(run(f));
  });
  test("unsafe destination syntax is rejected before filesystem mutation", () => {
    for (const dest of ["relative", "/", "/tmp/", "/tmp/.", "/tmp/.."]) {
      const f = fixture(); f.dest = dest; failure(run(f));
      assert.equal(fs.existsSync(f.log), false);
    }
  });
  test("bad archive hash never publishes a partial runtime", () => {
    const f = fixture(); failure(run(f, "", {NV_TEST_BAD_ARCHIVE: "1"}));
    assert.equal(fs.existsSync(f.dest), false); noTemporaryFiles(f);
  });
  test("transport failure propagates even from an if-condition caller", () => {
    const f = fixture(); failure(run(f, "", {NV_TEST_FAIL: "1"}, true));
    assert.equal(fs.existsSync(f.dest), false); noTemporaryFiles(f);
  });
  test("a conflicting concurrent creator cannot be overwritten", () => {
    const f = fixture(); failure(run(f, "", {NV_TEST_RACE: "1"}));
    assert.equal(fs.readFileSync(path.join(f.dest, "foreign"), "utf8"), "keep");
    assert.deepEqual(fs.readdirSync(f.dest), ["foreign"]); noTemporaryFiles(f);
  });
  test("two legitimate concurrent installers converge on one verified tree", () => {
    const f = fixture();
    const result = spawnSync("bash", ["-c", 'source "$1"; install_nvrtc_current "$2" & first=$!; '
      + 'install_nvrtc_current "$2" & second=$!; wait "$first"; a=$?; wait "$second"; b=$?; '
      + 'test "$a" = 0 && test "$b" = 0', "race", helper, f.dest],
    {env: env(f), encoding: "utf8", timeout: 60000});
    assert.ifError(result.error); success(result);
    success(run(f, "--validate-only")); noTemporaryFiles(f);
  });
  test("a skipped publication status requires the complete destination to verify", () => {
    const good = fixture(); success(run(good, "", {NV_TEST_MOVE_STATUS: "valid"}));
    success(run(good, "--validate-only")); noTemporaryFiles(good);
    const bad = fixture(); failure(run(bad, "", {NV_TEST_MOVE_STATUS: "bad"}));
    assert.equal(fs.readFileSync(path.join(bad.dest, "foreign"), "utf8"), "keep");
    noTemporaryFiles(bad);
  });
  test("development setup wires the shared compiler without an older NVRTC package", () => {
    const source = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
    assert.match(source, /\. "\$SCRIPT_DIR\/install-nvrtc-current\.sh"/u);
    assert.match(source, /apt_install "cuda-minimal-build-\$\{CUDA_VERSION\}"/u);
    assert.doesNotMatch(source, /apt_install[^\n]*cuda-nvrtc-dev/u);
    assert.match(source, /ln -s \/opt\/mom\/nvrtc-13\.3\.33\/include\/nvrtc\.h/u);
  });
  test("CUDA SDK readiness rejects missing NVML headers before a native runtime build", () => {
    const source = fs.readFileSync(path.join(__dirname, "../scripts/install-dev.sh"), "utf8");
    const start = source.indexOf("cuda_payload_ready() {"), end = source.indexOf("\n}\n", start);
    assert.ok(start >= 0 && end > start);
    const root = fs.mkdtempSync(path.join(work, "cuda-payload-"));
    const files = ["bin/nvcc", "bin/ptxas", "nvvm/libdevice/libdevice.10.bc", "include/nvrtc.h", "include/cuda/std/cstdint", "include/cuda.h", "include/nvml.h"];
    for (const name of files) {const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture", {mode: 0o755});}
    const check = () => spawnSync("bash", ["-c", source.slice(start, end + 2) + '\ncuda_payload_ready "$1"', "test", root], {encoding: "utf8"});
    assert.equal(check().status, 0);
    for (const name of ["nvml.h", "cuda.h"]) {
      const file = path.join(root, "include", name); fs.unlinkSync(file);
      assert.equal(check().status, 1, "Incomplete CUDA SDK must reject missing " + name);
      fs.writeFileSync(file, "fixture");
    }
    assert.match(source, /apt_install[^\n]*"cuda-nvml-dev-\$\{CUDA_VERSION\}"/u);
  });
  test("runtime setup separates shared provisioning from standalone legacy fallback", () => {
    const source = fs.readFileSync(path.join(__dirname, "../scripts/install.sh"), "utf8");
    assert.match(source, /if ! declare -F install_nvrtc_current >\/dev\/null; then\n\s+packages\+=\(libnvrtc12\)/u);
    assert.match(source, /install_nvrtc_current\n\s+else/u);
    assert.match(source, /ca-certificates curl g\+\+ xz-utils/u);
  });
  test("Engineering consumers use the shared helper and native release CI admits pinned SDKs", () => {
    for (const relative of ["scripts/build-combined.dockerfile", "scripts/build-multicompiler.dockerfile",
      "scripts/build-nvidia-adaptivecpp.dockerfile", ".github/workflows/scripts/package-linux-combined.sh"]) {
      assert.match(fs.readFileSync(path.join(__dirname, "..", relative), "utf8"), /install-nvrtc-current\.sh/u);
    }
    assert.match(fs.readFileSync(path.join(__dirname, "run-hash.js"), "utf8"), /"tests\/nvrtc-installer\.js"/u);

    const releaseWorkflow = fs.readFileSync(path.join(__dirname, "../.github/workflows/build-release-artifacts.yml"), "utf8");
    assert.match(releaseWorkflow, /uses: \.\/\.github\/workflows\/build-native-prerequisites\.yml/u);
    assert.doesNotMatch(releaseWorkflow, /install-nvrtc-current\.sh/u);
    const nativeWorkflow = fs.readFileSync(path.join(__dirname, "../.github/workflows/build-native-prerequisites.yml"), "utf8");
    assert.match(nativeWorkflow, /^ {2}verify-linux-nvrtc:/mu);
    assert.match(nativeWorkflow, /^ {2}verify-windows-nvrtc:/mu);
    assert.match(nativeWorkflow, /node scripts\/prepare-native-ci\.js stage nvrtc-linux/u);
    assert.match(nativeWorkflow, /node scripts\/prepare-native-ci\.js stage nvrtc-windows/u);
    assert.match(nativeWorkflow, /tests\/native-nvrtc-sdk\.js/u);
  });
});
after(() => {if (work) {fs.rmSync(work, {recursive: true, force: true});}});
