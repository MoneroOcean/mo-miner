"use strict";

const assert = require("node:assert/strict"), test = require("node:test"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const sysroot = require("../scripts/native-linux-sysroot");
const {sysrootArchiveCatalog, libdrmArchiveCatalog} = require("../scripts/fetch-native-sdk-sources");
const {libdrmPin} = require("../scripts/build-native-libdrm");
const {nativeSdkEnvironment} = require("../scripts/build-environment");

// A real prepared sysroot is opt-in: point MOM_TEST_SYSROOT_PACKAGES at a verified `fetch-native-sdk-sources.js sysroot` output.
const packages = process.env["MOM_TEST_SYSROOT_PACKAGES"];

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

  test("ambient Intel compiler configuration, pkg-config and toolchain variables are rejected, not inherited", () => {
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

  test("non-Intel drivers receive the sysroot and its focal GCC 14 installation explicitly", () => {
    assert.deepEqual(sysroot.linuxSysrootFlags({root: "/s/root", gccInstallDir: "/s/root/usr/lib/gcc/x86_64-linux-gnu/14"}),
      ["--sysroot=/s/root", "--gcc-install-dir=/s/root/usr/lib/gcc/x86_64-linux-gnu/14"]);
  });

  test("the tree digest covers paths, contents and link targets and rejects absolute links", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sysroot-digest-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
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
    assert.match(fs.readFileSync(prepared.toolchain, "utf8"), /set\(CMAKE_SYSROOT "[^"]+"\)\nlist\(APPEND CMAKE_IGNORE_PATH \/usr\/include /);
    assert.doesNotMatch(fs.readFileSync(prepared.toolchain, "utf8"), /CMAKE_SYSTEM_NAME/, "a sysroot build is not a cross build");
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
