"use strict";

const assert = require("node:assert/strict"), test = require("node:test"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const sysroot = require("../scripts/native-linux-sysroot");
const {sysrootArchiveCatalog, libdrmArchiveCatalog} = require("../scripts/fetch-native-sdk-sources");
const {libdrmPin} = require("../scripts/build-native-libdrm");
const {nativeSdkEnvironment} = require("../scripts/build-environment");
const {tempDir} = require("./common/temp-dir");

// A real prepared sysroot is opt-in: point MOM_TEST_SYSROOT_PACKAGES at a verified `fetch-native-sdk-sources.js sysroot` output.
const packages = process.env["MOM_TEST_SYSROOT_PACKAGES"];

/** Configure a tiny CMake project with the generated toolchain and report what find_* resolved.
 * The host C compiler is enough: the probe only exercises CMake's search policy.
 * @param {import("node:test").TestContext} t @param {{root: string, toolchain: string}} prepared */
function assertToolchainSearch(t, prepared) {
  const work = fs.realpathSync(tempDir(t, "mom-sysroot-cmake-"));
  const sdk = path.join(path.dirname(path.dirname(prepared.root)), "fixture-sdk");
  fs.mkdirSync(path.join(sdk, "lib"), {recursive: true}); fs.writeFileSync(path.join(sdk, "lib/libfixture.a"), "!<arch>\n");
  fs.writeFileSync(path.join(work, "CMakeLists.txt"), ["cmake_minimum_required(VERSION 3.20)", "project(probe C)",
    "find_library(PROBE_LIBC c)", "find_path(PROBE_STDIO stdio.h)", "find_library(PROBE_NUMA numa)",
    "find_path(PROBE_ZLIB zlib.h)", "find_library(PROBE_LIBZ z)", "find_library(PROBE_SDK fixture HINTS " + path.join(sdk, "lib") + ")",
    "find_program(PROBE_CMAKE cmake)",
    // eslint-disable-next-line no-template-curly-in-string -- CMake, not JavaScript, expands these variables.
    'foreach(name LIBC STDIO NUMA ZLIB LIBZ SDK CMAKE)\n  message(STATUS "PROBE_${name}=${PROBE_${name}}")\nendforeach()', ""].join("\n"));
  const result = require("node:child_process").spawnSync("cmake", ["-S", work, "-B", path.join(work, "build"),
    "-DCMAKE_TOOLCHAIN_FILE=" + prepared.toolchain], {encoding: "utf8", timeout: 120000});
  assert.equal(result.status, 0, result.stderr);
  const found = Object.fromEntries([...result.stdout.matchAll(/PROBE_([A-Z]+)=(.*)/g)].map(match => [match[1], match[2]]));
  const lib = path.join(prepared.root, "usr/lib/x86_64-linux-gnu");
  assert.equal(found["LIBC"], path.join(lib, "libc.so"), "system libraries resolve inside the sysroot");
  assert.equal(found["STDIO"], path.join(prepared.root, "usr/include"));
  assert.equal(found["NUMA"], path.join(lib, "libnuma.a"));
  assert.equal(found["ZLIB"], "PROBE_ZLIB-NOTFOUND", "host-only headers are not visible");
  assert.equal(found["LIBZ"], "PROBE_LIBZ-NOTFOUND", "host-only libraries are not visible");
  assert.equal(found["SDK"], path.join(sdk, "lib/libfixture.a"), "SDK outputs beside the sysroot stay searchable");
  assert.ok(path.isAbsolute(found["CMAKE"] || "") && !(found["CMAKE"] || "").startsWith(prepared.root), "programs still come from the host");
}

test.describe("glibc 2.31 Linux sysroot", () => {
  test("the baseline is the single glibc 2.31 floor and every pin is a mirrored focal/PPA package", () => {
    assert.equal(sysroot.linuxGlibcBaseline, "2.31");
    assert.deepEqual(sysroot.linuxSysrootPackages.map(row => row.package).sort(), ["libc6", "libc6-dev", "libcrypt-dev",
      "libelf-dev", "libgcc-14-dev", "libgcc-s1", "libnuma-dev", "libstdc++-14-dev", "linux-libc-dev"]);
    for (const row of sysroot.linuxSysrootPackages) {
      assert.match(row.sha256, /^[0-9a-f]{64}$/);
      assert.match(row.asset, /^[a-z0-9.+-]+_[A-Za-z0-9.+-]+_amd64\.deb$/);
      assert.doesNotMatch(row.asset, /[~+]/, "mirror asset names spell ~ as . and libstdc++ as libstdcxx");
    }
  });

  test("fetches only from the pinned mirror release, keyed by package stem", () => {
    const catalog = sysrootArchiveCatalog();
    assert.equal(Object.keys(catalog).length, sysroot.linuxSysrootPackages.length);
    for (const [key, row] of Object.entries(catalog)) {
      assert.equal(row.url, sysroot.linuxSysrootRelease + row.fileName);
      assert.equal(key, row.fileName.split("_")[0]);
    }
    assert.ok(sysroot.linuxSysrootRelease.startsWith("https://github.com/MoneroOcean/mo-miner/releases/download/sysroot-inputs-v1/"));
    assert.deepEqual(libdrmArchiveCatalog().libdrm, {fileName: libdrmPin.fileName, sha256: libdrmPin.sha256, url: libdrmPin.url});
    assert.throws(() => sysrootArchiveCatalog({libc6: {sha256: "0".repeat(64)}}), /exact pinned package cohort/);
  });

  test("the receipt identity is location-independent and changes with any pin", () => {
    const text = JSON.stringify(sysroot.linuxSysrootReceipt(), null, 2) + "\n";
    assert.equal(require("node:crypto").createHash("sha256").update(text).digest("hex"), sysroot.linuxSysrootReceiptSha256);
    assert.doesNotMatch(text, /\/tmp|\/home/);
    assert.equal(sysroot.linuxSysrootReceipt().treeSha256, sysroot.linuxSysrootTreeSha256);
  });

  test("producer receipts must name the current sysroot receipt", () => {
    const record = sysroot.linuxSysrootRecord({receiptSha256: sysroot.linuxSysrootReceiptSha256});
    sysroot.verifyLinuxSysrootRecord(record, "Fixture SDK");
    assert.throws(() => sysroot.verifyLinuxSysrootRecord(undefined, "Fixture SDK"), /Fixture SDK was not built against the pinned glibc 2\.31 sysroot/);
    assert.throws(() => sysroot.verifyLinuxSysrootRecord({...record, receiptSha256: "0".repeat(64)}, "Fixture SDK"), /rebuild it/);
  });

  test("ambient Intel compiler configuration, pkg-config and toolchain variables are rejected, not inherited", {skip: process.platform !== "linux"}, () => {
    for (const name of ["ICXCFG", "ICPXCFG", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_SYSROOT_DIR"]) {
      assert.throws(() => sysroot.applyLinuxSysrootEnvironment({PATH: "/usr/bin", [name]: "/host"}), new RegExp("Unset " + name));
      assert.throws(() => nativeSdkEnvironment({PATH: "/usr/bin", [name]: "/host"}), new RegExp("Unset " + name));
    }
    assert.throws(() => sysroot.applyLinuxSysrootEnvironment({CMAKE_TOOLCHAIN_FILE: "/host.cmake"}), /Unset CMAKE_TOOLCHAIN_FILE/);
    assert.deepEqual(sysroot.applyLinuxSysrootEnvironment({PATH: "/usr/bin"}), {PATH: "/usr/bin"});
    assert.throws(() => sysroot.applyLinuxSysrootEnvironment({MOM_LINUX_SYSROOT_SHA256: "0".repeat(64)}), /does not identify/);
    assert.throws(() => sysroot.requireLinuxSysroot({}), /require MOM_LINUX_SYSROOT/);
    assert.equal(sysroot.linuxSysrootToolchain({}), undefined);
    assert.equal(sysroot.linuxSysrootToolchain({MOM_LINUX_SYSROOT: "/work/sysroot"}), "/work/sysroot/toolchain.cmake");
  });

  test("tool identity text does not depend on whether the sysroot's compiler configuration is selected", () => {
    const plain = "Intel(R) oneAPI DPC++/C++ Compiler 2026.2.0\nTarget: x86_64-unknown-linux-gnu\nInstalledDir: /opt/intel/bin\n";
    assert.equal(sysroot.compilerIdentityText(plain + "Configuration file: /work/sysroot/compiler.cfg\n"), plain);
    assert.equal(sysroot.compilerIdentityText(plain), plain);
  });

  test("non-Intel drivers receive the sysroot and its focal GCC 14 installation explicitly", () => {
    assert.deepEqual(sysroot.linuxSysrootFlags({root: "/s/root", gccInstallDir: "/s/root/usr/lib/gcc/x86_64-linux-gnu/14"}),
      ["--sysroot=/s/root", "--gcc-install-dir=/s/root/usr/lib/gcc/x86_64-linux-gnu/14"]);
  });

  test("the tree digest covers paths, contents and link targets and rejects absolute links", t => {
    const root = tempDir(t, "mom-sysroot-digest-");
    fs.mkdirSync(path.join(root, "usr/lib"), {recursive: true});
    fs.writeFileSync(path.join(root, "usr/lib/libc.so"), "GROUP ( /lib/libc.so.6 )\n");
    fs.symlinkSync("../../lib/libc.so.6", path.join(root, "usr/lib/libc.so.6"));
    const first = sysroot.treeDigest(root);
    assert.equal(first.entries, 4);
    assert.deepEqual(sysroot.treeDigest(root), first);
    fs.writeFileSync(path.join(root, "usr/lib/libc.so"), "changed\n");
    assert.notEqual(sysroot.treeDigest(root).sha256, first.sha256);
    fs.symlinkSync("/lib/libdl.so.2", path.join(root, "usr/lib/libdl.so"));
    assert.throws(() => sysroot.treeDigest(root), /absolute symlink/);
  });

  test("prepares the pinned tree, compiler configuration and CMake toolchain", {skip: !packages || process.platform !== "linux"}, t => {
    const parent = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-sysroot-"))), output = path.join(parent, "sysroot");
    t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
    const prepared = sysroot.prepareLinuxSysroot(packages || "", output);
    assert.equal(prepared.receiptSha256, sysroot.linuxSysrootReceiptSha256);
    assert.equal(fs.readFileSync(prepared.compilerConfig, "utf8"), `--sysroot=${prepared.root}\n--gcc-install-dir=${prepared.gccInstallDir}\n`);
    assertToolchainSearch(t, prepared);
    for (const file of ["usr/lib/gcc/x86_64-linux-gnu/14/libstdc++.a", "usr/lib/x86_64-linux-gnu/libelf.a", "usr/lib/x86_64-linux-gnu/libnuma.a",
      "usr/share/doc/libelf-dev/copyright"]) {
      assert.ok(fs.statSync(path.join(prepared.root, file)).isFile(), file);
    }
    const environment = sysroot.applyLinuxSysrootEnvironment({MOM_LINUX_SYSROOT: output});
    assert.equal(environment["ICPXCFG"], prepared.compilerConfig);
    assert.equal(environment["CMAKE_TOOLCHAIN_FILE"], prepared.toolchain);
    assert.deepEqual(sysroot.applyLinuxSysrootEnvironment(environment), environment, "an applied environment stays idempotent");
    fs.appendFileSync(path.join(prepared.root, "usr/include/stdio.h"), "\n");
    assert.throws(() => require("node:child_process").execFileSync(process.execPath, [path.join(__dirname, "../scripts/native-linux-sysroot.js"), "audit", output],
      {stdio: "pipe"}), /tree changed/);
  });
});
