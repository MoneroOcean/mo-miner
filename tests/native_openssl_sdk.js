"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {fileHash} = require("../scripts/build-source-inputs");
const {opensslPin, configureOptions, toolNames, opensslSdkPlan, opensslSdkEnvironment, validateArchive, opensslBuildRecipe, nativeOpenSslInputs, buildNativeOpenSsl, stageWindowsHeaders} = require("../scripts/build-native-openssl");

const linuxTools = {ar: "/tools/ar", cc: "/tools/clang", make: "/tools/make", perl: "/tools/perl", ranlib: "/tools/ranlib", tar: "/tools/tar"};
const windowsTools = {ar: "/tools/lib.exe", cc: "/tools/cl.exe", link: "/tools/link.exe", make: "/tools/nmake.exe", perl: "/tools/perl.exe", rc: "/tools/rc.exe", tar: "/tools/tar.exe"};

/** Temporary fixtures never execute a compiler, OpenSSL or miner. @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-openssl-")); t.after(() => fs.rmSync(root, {recursive: true, force: true})); return root;
}

/** @param {import("node:test").TestContext} t */
function fakeTools(t) {
  const root = fixture(t), tools = Object.fromEntries(toolNames("linux").map(name => {
    const file = path.join(root, name); fs.writeFileSync(file, "#!/bin/sh\nexit 0\n"); fs.chmodSync(file, 0o755); return [name, file];
  }));
  return {root, tools};
}

/** Build a receipt-shaped SDK from inert bytes; no output program is ever executed.
 * @param {import("node:test").TestContext} t @param {"linux" | "win32"} platform */
function sdkFixture(t, platform) {
  const root = fixture(t), extension = platform === "win32" ? ".lib" : ".a", executable = platform === "win32" ? "openssl.exe" : "openssl";
  const license = path.join(root, "source", "LICENSE.txt"), header = path.join(root, "include", "openssl", "opensslv.h");
  fs.mkdirSync(path.dirname(license), {recursive: true}); fs.writeFileSync(license, "OpenSSL license notice\n");
  fs.mkdirSync(path.dirname(header), {recursive: true}); fs.writeFileSync(header, '#define OPENSSL_VERSION_STR "3.5.9"\n');
  fs.mkdirSync(path.join(root, "lib")); fs.mkdirSync(path.join(root, "bin"));
  fs.writeFileSync(path.join(root, "lib", "libcrypto" + extension), "!<arch>\n");
  fs.writeFileSync(path.join(root, "lib", "libssl" + extension), "!<arch>\n");
  const binary = Buffer.alloc(256);
  if (platform === "win32") {binary.write("MZ"); binary.writeUInt32LE(128, 0x3c); binary.write("PE\0\0", 128, "binary");} else {binary.write("\x7fELF", 0, "binary");}
  const cli = path.join(root, "bin", executable); fs.writeFileSync(cli, binary); fs.chmodSync(cli, 0o755);
  const artifacts = Object.fromEntries([header, path.join(root, "lib", "libcrypto" + extension), path.join(root, "lib", "libssl" + extension), cli]
    .map(file => [path.relative(root, file).split(path.sep).join("/"), fileHash(file)]));
  const tools = Object.fromEntries(toolNames(platform).map(name => [name, {path: "/tools/" + name, sha256: "a".repeat(64)}]));
  const plan = opensslSdkPlan(root, Object.fromEntries(Object.entries(tools).map(([name, row]) => [name, row.path])), 1, platform);
  const record = {version: 1, opensslVersion: opensslPin.version, platform, target: platform === "win32" ? "VC-WIN64A" : "linux-x86_64",
    archive: {path: "/source/openssl-3.5.9.tar.gz", sha256: opensslPin.sha256, bytes: 1024, members: 3},
    source: {files: 1, bytes: fs.statSync(license).size, sha256: "b".repeat(64), notice: {path: "source/LICENSE.txt", sha256: fileHash(license)}},
    sourceDateEpoch: plan.sourceDateEpoch, configure: plan.configure, config: {sha256: "c".repeat(64)}, tools, recipe: opensslBuildRecipe(), artifacts,
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(record));
  return {root, record, save: () => fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(record))};
}

test.describe("Pinned static OpenSSL SDK production and admission", () => {
  test("the pinned source and lean static policy match the TLS runtime contract", () => {
    assert.equal(opensslPin.version, "3.5.9"); assert.equal(opensslPin.sha256, "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a");
    assert.equal(opensslPin.sourceDateEpoch, 1790640000);
    assert.deepEqual(configureOptions, ["no-shared", "no-module", "no-dso", "no-engine", "no-tests", "no-comp", "no-zlib", "no-zstd", "no-brotli"]);
    assert.ok(Object.isFrozen(opensslPin)); assert.ok(Object.isFrozen(configureOptions));
  });

  test("Linux uses explicit compiler/archive tools and the requested bounded GNU Make jobs", () => {
    for (const jobs of [1, 2, 16]) {
      const plan = opensslSdkPlan("/new/openssl", linuxTools, jobs, "linux");
      assert.equal(plan.configure[2], "linux-x86_64"); assert.deepEqual(plan.buildCommand.slice(0, 3), [linuxTools.make, "-j", String(jobs)]);
      assert.deepEqual(plan.installCommand.slice(0, 3), [linuxTools.make, "-j", String(jobs)]); assert.equal(plan.parallelJobs, jobs);
      for (const option of configureOptions) {assert.ok(plan.configure.includes(option));}
      assert.equal(plan.toolEnvironment.CC, linuxTools.cc); assert.equal(plan.toolEnvironment.AR, linuxTools.ar);
      assert.equal(plan.toolEnvironment.RANLIB, linuxTools.ranlib); assert.equal(plan.toolEnvironment.LD, linuxTools.cc);
      assert.match(plan.libraries.crypto.replace(/\\/g, "/"), /lib\/libcrypto\.a$/);
      assert.match(plan.executable.replace(/\\/g, "/"), /bin\/openssl$/);
    }
  });

  test("Linux config uses stable logical paths, private DESTDIR layout and fixed source epoch", () => {
    const first = opensslSdkPlan("/new/openssl-a", linuxTools, 2, "linux"), second = opensslSdkPlan("/new/openssl-b", linuxTools, 2, "linux");
    const logicalArgs = (/** @type {ReturnType<typeof opensslSdkPlan>} */ plan) => plan.configure.filter(arg => arg.startsWith("--prefix=") || arg.startsWith("--openssldir="));
    assert.deepEqual(logicalArgs(first), ["--prefix=/usr/local", "--openssldir=/etc/ssl"]);
    assert.deepEqual(logicalArgs(second), logicalArgs(first));
    assert.equal(first.sourceDateEpoch, 1790640000);
    assert.equal(opensslSdkEnvironment({SOURCE_DATE_EPOCH: "1"}, first)["SOURCE_DATE_EPOCH"],
      opensslSdkEnvironment({SOURCE_DATE_EPOCH: "2"}, second)["SOURCE_DATE_EPOCH"]);
    assert.equal(first.install, path.join("/new/openssl-a", "install"));
    assert.equal(first.stagedPrefix, path.join(first.install, "usr", "local"));
    assert.deepEqual(first.installCommand, [linuxTools.make, "-j", "2", "DESTDIR=" + first.install, "install_dev"]);
    assert.equal(first.installCommand.includes("DESTDIR=/new/openssl-b/install"), false);
  });

  test("OpenSSL child environment overrides caller epoch and cannot redirect staging", () => {
    const plan = opensslSdkPlan("/new/openssl", linuxTools, 2, "linux"), caller = {
      DESTDIR: "/caller/redirect", source_date_epoch: "1", SOURCE_DATE_EPOCH: "2", MAKEFLAGS: "-e",
      makefiles: "/caller/injected.mk", MAKEOVERRIDES: "DESTDIR=/caller/redirect",
    };
    const child = opensslSdkEnvironment(caller, plan);
    assert.equal(child["DESTDIR"], undefined); assert.equal(child["source_date_epoch"], undefined);
    assert.equal(child["SOURCE_DATE_EPOCH"], String(plan.sourceDateEpoch));
    assert.equal(child["MAKEFLAGS"], undefined); assert.equal(child["makefiles"], undefined); assert.equal(child["MAKEOVERRIDES"], undefined);
    assert.equal(caller.DESTDIR, "/caller/redirect"); assert.equal(caller.SOURCE_DATE_EPOCH, "2");
    assert.equal(plan.installCommand.includes("DESTDIR=" + plan.install), true);
  });

  test("one recursive OpenSSL goal owns library and application dependencies", () => {
    for (const jobs of [2, 16]) {
      const linux = opensslSdkPlan("/new/openssl", linuxTools, jobs, "linux");
      assert.deepEqual(linux.buildCommand.slice(3), ["build_sw"]);
    }
    const windows = opensslSdkPlan("/new/openssl", windowsTools, 2, "win32");
    assert.deepEqual(windows.buildCommand.slice(2), ["build_sw"]);
  });

  test("Windows uses VC-WIN64A static CRT policy, static archive names and a build-only PE CLI", () => {
    const plan = opensslSdkPlan("/new/openssl", windowsTools, 4, "win32");
    assert.equal(plan.configure[2], "VC-WIN64A"); assert.ok(plan.configure.includes("no-shared")); assert.ok(plan.configure.includes("no-asm"));
    assert.equal(plan.configure.some(value => value === "--MT"), false); assert.equal(plan.parallelJobs, 1);
    assert.deepEqual(plan.buildCommand, [windowsTools.make, "/NOLOGO", "build_sw"]);
    assert.deepEqual(plan.installCommand, []); // Intel emits CodeView in objects, not MSVC's ossl_static.pdb.
    assert.deepEqual(plan.configure.filter(value => value.startsWith("--prefix=") || value.startsWith("--openssldir=")),
      ["--prefix=C:/OpenSSL", "--openssldir=C:/OpenSSL/ssl"]);
    assert.deepEqual(opensslSdkPlan("/other/output", windowsTools, 1, "win32").configure.filter(value =>
      value.startsWith("--prefix=") || value.startsWith("--openssldir=")), ["--prefix=C:/OpenSSL", "--openssldir=C:/OpenSSL/ssl"]);
    assert.equal(plan.stagedPrefix, null); assert.equal(plan.sourceDateEpoch, 1790640000);
    assert.match(plan.libraries.ssl, /libssl\.lib$/); assert.match(plan.buildExecutable.replace(/\\/g, "/"), /build\/apps\/openssl\.exe$/);
    assert.equal(plan.toolEnvironment.CC, "/tools/cl.exe"); assert.equal(plan.toolEnvironment.LD, "/tools/link.exe");
  });

  test("Windows leaves tool paths unquoted for OpenSSL to quote exactly once", () => {
    const perl = "/SDK Tools/Perl/bin/perl.exe", tools = Object.fromEntries(Object.entries(windowsTools).map(([key, file]) => [key, file.replace("/tools/", "/SDK Tools/")]));
    tools["perl"] = perl;
    const plan = opensslSdkPlan("/new/openssl", tools, 2, "win32");
    assert.equal(plan.configure[0], perl);
    assert.deepEqual(plan.toolEnvironment, {PERL: perl, CC: tools["cc"], AR: tools["ar"], LD: tools["link"], RC: tools["rc"]});
    assert.equal(opensslSdkPlan("/new/openssl", windowsTools, 2, "win32").toolEnvironment.PERL, windowsTools.perl);
  });

  test("Windows stages the public source/generated header union without installing PDBs", t => {
    const root = fixture(t), source = path.join(root, "source"), build = path.join(root, "build"), target = path.join(root, "headers");
    for (const directory of [source, build]) {fs.mkdirSync(path.join(directory, "include/openssl"), {recursive: true});}
    fs.writeFileSync(path.join(source, "include/openssl/ssl.h"), "original public header");
    fs.writeFileSync(path.join(source, "include/openssl/ssl.h.in"), "must not be staged");
    fs.writeFileSync(path.join(build, "include/openssl/configuration.h"), "generated target configuration");
    stageWindowsHeaders(source, build, target);
    assert.deepEqual(fs.readdirSync(target).sort(), ["configuration.h", "ssl.h"]);
    assert.equal(fs.readFileSync(path.join(target, "ssl.h"), "utf8"), "original public header");
    assert.equal(fs.readFileSync(path.join(target, "configuration.h"), "utf8"), "generated target configuration");
    assert.ok(!fs.existsSync(path.join(root, "ossl_static.pdb")));
    assert.throws(() => stageWindowsHeaders(source, build, target), /EEXIST/);
  });

  test("Windows public header staging rejects a non-file .h instead of following it", t => {
    const root = fixture(t), source = path.join(root, "source"), build = path.join(root, "build");
    for (const directory of [source, build]) {fs.mkdirSync(path.join(directory, "include/openssl"), {recursive: true});}
    fs.mkdirSync(path.join(source, "include/openssl/redirect.h"));
    assert.throws(() => stageWindowsHeaders(source, build, path.join(root, "headers")), /regular public headers/);
  });

  test("invalid plans and implicit tool lookup are rejected", () => {
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => opensslSdkPlan("/new/sdk", linuxTools, jobs), /jobs/);}
    for (const output of ["relative", "/new/bad;path", "/new/$PATH", "/new/line\nbreak"]) {assert.throws(() => opensslSdkPlan(output, linuxTools, 2), /absolute output/);}
    assert.throws(() => opensslSdkPlan("/new/sdk", {...linuxTools, cc: "clang"}, 2), /explicit absolute/);
    const missing = {...linuxTools}; Reflect.deleteProperty(missing, "tar");
    assert.throws(() => opensslSdkPlan("/new/sdk", missing, 2), /explicit absolute/);
    assert.throws(() => opensslSdkPlan("/new/sdk", linuxTools, 2, "darwin"), /Linux\/Windows/);
  });

  test("archive validation rejects traversal, links and incomplete original roots", () => {
    const good = ["openssl-3.5.9/", "openssl-3.5.9/Configure", "openssl-3.5.9/LICENSE.txt"].join("\n") + "\n";
    const modes = ["drwxr-xr-x root/root 0 2025-01-01 00:00 openssl-3.5.9/", "-rw-r--r-- root/root 1 2025-01-01 00:00 Configure", "-rw-r--r-- root/root 1 2025-01-01 00:00 LICENSE.txt"].join("\n") + "\n";
    assert.equal(validateArchive(good, modes), 3);
    const invalidCases = /** @type {[string, string][]} */ ([
      [good + "openssl-3.5.9/../../escape\n", modes + "-rw-r--r-- root/root 1 2025-01-01 00:00 escape\n"],
      [good + "openssl-3.5.9/link\n", modes + "lrwxrwxrwx root/root 0 2025-01-01 00:00 link -> target\n"],
      [good.replace("openssl-3.5.9/LICENSE.txt\n", ""), modes.replace(/-rw-r--r-- root\/root 1 2025-01-01 00:00 LICENSE.txt\n$/, "")],
      [good.replace("openssl-3.5.9/", "other/") , modes],
    ]);
    for (const [listing, verbose] of invalidCases) {assert.throws(() => validateArchive(listing, verbose), /archive|member|Configure|license/i);}
  });

  test("the build-only boundary and invalid preflight fail before output creation", t => {
    const root = fixture(t), output = path.join(root, "new-sdk");
    assert.throws(() => buildNativeOpenSsl("/missing.tar.gz", output, {version: 1, jobs: 2, tools: linuxTools}, {}), /MOM_AOT_BUILD_ONLY=1/);
    assert.equal(fs.existsSync(output), false);
    assert.throws(() => buildNativeOpenSsl("relative.tar.gz", output, {version: 1, jobs: 2, tools: linuxTools}, {MOM_AOT_BUILD_ONLY: "1"}), /absolute/);
    assert.equal(fs.existsSync(output), false);
  });

  test("bad archive bytes fail before mutation even with valid explicit tool files", {skip: process.platform !== "linux"}, t => {
    const f = fakeTools(t), archive = path.join(f.root, "wrong.tar.gz"), output = path.join(f.root, "new-sdk");
    fs.writeFileSync(archive, "not the pinned source archive");
    const config = {version: 1, jobs: 2, tools: f.tools};
    assert.throws(() => buildNativeOpenSsl(archive, output, config, {MOM_AOT_BUILD_ONLY: "1"}), /SHA256/);
    assert.equal(fs.existsSync(output), false);
  });

  test("source archive symlinks and output overwrites are rejected without touching the target", {skip: process.platform === "win32"}, t => {
    const f = fakeTools(t), archive = path.join(f.root, "source.tar.gz"), alias = path.join(f.root, "alias.tar.gz"), output = path.join(f.root, "new-sdk");
    fs.writeFileSync(archive, "not the pinned archive"); fs.symlinkSync(archive, alias);
    const config = {version: 1, jobs: 2, tools: f.tools};
    assert.throws(() => buildNativeOpenSsl(alias, output, config, {MOM_AOT_BUILD_ONLY: "1"}), /canonical regular file/);
    assert.equal(fs.existsSync(output), false);
    fs.mkdirSync(output); fs.writeFileSync(path.join(output, "keep"), "unchanged");
    assert.throws(() => buildNativeOpenSsl(archive, output, config, {MOM_AOT_BUILD_ONLY: "1"}), /new, canonical/);
    assert.equal(fs.readFileSync(path.join(output, "keep"), "utf8"), "unchanged");
  });

  for (const platform of ["linux", "win32"]) {
    test(`${platform} SDK admission verifies exact staged headers, archives, CLI and notice`, {skip: platform === "linux" && process.platform !== "linux"}, t => {
      const f = sdkFixture(t, /** @type {"linux" | "win32"} */ (platform)), result = nativeOpenSslInputs(f.root, platform);
      assert.equal(result.headers, path.join(f.root, "include", "openssl")); assert.equal(result.libraries.length, 2);
      assert.equal(result.executable, path.join(f.root, "bin", platform === "win32" ? "openssl.exe" : "openssl"));
      assert.equal(Object.keys(result.artifacts).length, 4);
      const license = path.join(f.root, "source", "LICENSE.txt");
      assert.deepEqual(result.notices, [{component: "OpenSSL LICENSE", path: license, sha256: fileHash(license)}]);
      assert.equal(result.receipt, path.join(f.root, "result.json"));
      assert.equal(result.files[result.receipt], result.receiptSha256);
      for (const relative of [...Object.keys(result.artifacts), "source/LICENSE.txt", "result.json"]) {
        const file = path.join(f.root, relative);
        assert.equal(result.files[file], fileHash(file));
      }
    });
  }

  test("changed or unrecorded SDK files cannot be admitted", t => {
    const f = sdkFixture(t, "linux"); fs.appendFileSync(path.join(f.root, "include", "openssl", "opensslv.h"), "changed\n");
    assert.throws(() => nativeOpenSslInputs(f.root, "linux"), /membership or bytes/);
    const g = sdkFixture(t, "linux"); fs.writeFileSync(path.join(g.root, "include", "openssl", "shadow.h"), "shadow\n");
    assert.throws(() => nativeOpenSslInputs(g.root, "linux"), /membership or bytes/);
  });

  test("receipt mutation cannot bless a changed source notice or a non-archive library", t => {
    const f = sdkFixture(t, "linux"); fs.appendFileSync(path.join(f.root, "source", "LICENSE.txt"), "changed\n");
    assert.throws(() => nativeOpenSslInputs(f.root, "linux"), /notice changed/);
    const g = sdkFixture(t, "linux"), archive = path.join(g.root, "lib", "libcrypto.a"); fs.writeFileSync(archive, "not an archive");
    g.record.artifacts["lib/libcrypto.a"] = fileHash(archive); g.save();
    assert.throws(() => nativeOpenSslInputs(g.root, "linux"), /not an archive/);
  });

  test("receipt admission pins the source epoch and logical Configure paths", t => {
    const f = sdkFixture(t, "linux"); f.record.sourceDateEpoch += 1; f.save();
    assert.throws(() => nativeOpenSslInputs(f.root, "linux"), /Invalid static OpenSSL SDK receipt/);
    const g = sdkFixture(t, "linux"); g.record.configure = g.record.configure.map(value => value === "--prefix=/usr/local" ? "--prefix=" + g.root : value); g.save();
    assert.throws(() => nativeOpenSslInputs(g.root, "linux"), /Configure|receipt/);
  });
});
