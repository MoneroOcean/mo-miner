"use strict";

const assert = require("node:assert/strict"), test = require("node:test"), fs = require("node:fs"), path = require("node:path");
const {fileHash} = require("../scripts/build-source-inputs");
const {libdrmPin, libdrmSdkPlan, nativeLibdrmInputs, buildNativeLibdrm} = require("../scripts/build-native-libdrm");
const {linuxSysrootRecord, linuxSysrootReceiptSha256, linuxSysrootRelease} = require("../scripts/native-linux-sysroot");
const {tempDir} = require("./common/temp-dir");

const tools = {cc: "/intel/bin/icx", meson: "/usr/bin/meson", ninja: "/usr/bin/ninja", tar: "/usr/bin/tar"};

/** A complete installed SDK fixture with a receipt bound to the current sysroot.
 * @param {import("node:test").TestContext} t */
function sdkFixture(t) {
  const root = fs.realpathSync(tempDir(t, "mom-libdrm-sdk-"));
  /** @type {Record<string, string>} */ const files = {};
  for (const [name, data] of Object.entries({"lib/libdrm.a": "!<arch>\n", "lib/libdrm_amdgpu.a": "!<arch>\n",
    "lib/pkgconfig/libdrm.pc": "Name: libdrm\n", "lib/pkgconfig/libdrm_amdgpu.pc": "Name: libdrm_amdgpu\n",
    "include/xf86drm.h": "drm", "include/libdrm/amdgpu.h": "amdgpu", "include/libdrm/amdgpu_drm.h": "uapi"})) {
    const file = path.join(root, "sdk", name);
    fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); files[name] = fileHash(file);
  }
  const record = {version: 1, libdrmVersion: libdrmPin.version, archive: {path: "/fetch/" + libdrmPin.fileName, sha256: libdrmPin.sha256,
    bytes: libdrmPin.bytes}, amdgpuIds: "/usr/share/libdrm/amdgpu.ids", sysroot: linuxSysrootRecord({receiptSha256: linuxSysrootReceiptSha256}),
  files, buildOnly: true, executed: false, publicationQualified: false};
  const save = () => fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(record)); save();
  return {root, record, save};
}

test.describe("Static libdrm SDK producer", {skip: process.platform !== "linux"}, () => {
  test("pins the upstream release in the sysroot mirror and builds only libdrm core and amdgpu", () => {
    assert.equal(libdrmPin.url, linuxSysrootRelease + libdrmPin.fileName);
    const plan = libdrmSdkPlan("/out", "/fetch/" + libdrmPin.fileName, tools, 4);
    assert.deepEqual(plan.compile, [tools.ninja, "-C", "/out/build", "-j", "4"]);
    for (const option of ["-Ddefault_library=static", "-Damdgpu=enabled", "-Dintel=disabled", "-Dradeon=disabled", "-Dtests=false",
      "-Dudev=false", "--prefix=/out/sdk", "--datadir=/usr/share"]) {
      assert.ok(plan.setup.includes(option), option);
    }
    assert.deepEqual(plan.install.slice(-2), ["--tags", "devel"], "the runtime ASIC table is never installed outside the SDK");
  });

  test("tools must be explicit absolute paths with Intel's icx as the compiler", () => {
    assert.throws(() => libdrmSdkPlan("/out", "/fetch/a.tar.xz", {...tools, cc: "/usr/bin/gcc"}, 2), /icx, meson, ninja and tar/);
    assert.throws(() => libdrmSdkPlan("/out", "/fetch/a.tar.xz", {...tools, meson: "meson"}, 2), /icx, meson, ninja and tar/);
    assert.throws(() => libdrmSdkPlan("/out; rm", "/fetch/a.tar.xz", tools, 2), /literal absolute paths/);
    assert.throws(() => libdrmSdkPlan("/out", "/fetch/a.tar.xz", tools, 17), /jobs in 1\.\.16/);
  });

  test("admission binds every installed file and the glibc sysroot record", t => {
    const f = sdkFixture(t), admitted = nativeLibdrmInputs(f.root);
    assert.equal(admitted.drm, path.join(f.root, "sdk/lib/libdrm.a"));
    assert.equal(admitted.pkgConfigDir, path.join(f.root, "sdk/lib/pkgconfig"));
    fs.appendFileSync(path.join(f.root, "sdk/include/xf86drm.h"), "changed");
    assert.throws(() => nativeLibdrmInputs(f.root), /installed files differ/);
  });

  test("an SDK built against another glibc is not admitted", t => {
    const f = sdkFixture(t);
    f.record.sysroot = {baseline: "2.31", receiptSha256: "0".repeat(64)}; f.save();
    assert.throws(() => nativeLibdrmInputs(f.root), /libdrm SDK was not built against the pinned glibc 2\.31 sysroot/);
  });

  test("production needs the build-only boundary and a new output", t => {
    const f = sdkFixture(t);
    assert.throws(() => buildNativeLibdrm("/fetch/a.tar.xz", path.join(f.root, "new"), tools, {}), /MOM_AOT_BUILD_ONLY=1/);
    assert.throws(() => buildNativeLibdrm("/fetch/a.tar.xz", f.root, tools, {MOM_AOT_BUILD_ONLY: "1"}), /new canonical directory/);
  });
});
