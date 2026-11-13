"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const {librarySearches, driverSearches, directoryTracker, auditDriverTrace} = require("../scripts/audit-native-driver-trace");
const {auditNativeTrace} = require("../scripts/native-audit");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const start = '100 execve("/mom", ["/mom"], ["HOME=/work"]) = 0';
const fork = "100 clone(child_stack=NULL, flags=SIGCHLD) = 101";
const worker = '101 execve("/mom", ["/mom", "test"], ["MOM_CLUSTER_WORKER=1"]) = 0';

// Model a trusted OS symlink without needing an installed GPU driver, readelf or privilege in CI.
function aliasAudit(opencl = false, vaPresent = false) {
  const driver = opencl ? "/usr/lib/libigdrcl.so" : "/usr/lib/libze_intel_gpu.so.1.2", support = "/usr/lib/libsupport.so.1";
  const closure = {roots: [{requested: opencl ? driver : "/usr/lib/libze_intel_gpu.so.1", resolved: driver}], lateRoots: [],
    defaultSearch: ["/usr/lib", "/lib"], libraries: [driver, support].map(file => ({file, soname: path.basename(file), needed: [], search: []}))};
  /** @type {{exports: unknown}} */
  const loaded = {exports: {}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../scripts/audit-native-driver-trace.js"), "utf8"), {
    module: loaded,
    /** @param {string} name */
    require(name) {
      if (name === "node:fs") {
        return {...fs,
          /** @param {import("node:fs").PathLike} file */
          existsSync(file) {return String(file).endsWith("/libva.so.2") ? vaPresent : fs.existsSync(file);},
        };
      }
      if (name === "./inspect-native-driver-deps") {
        return {inspectDrivers: () => closure,
          /** @param {string} file */
          systemFile(file) {return file.startsWith("/lib/") ? "/usr" + file : file;},
        };
      }
      return name === "./native-audit" ? require("../scripts/native-audit") : require(name);
    },
  });
  return /** @type {typeof import("../scripts/audit-native-driver-trace")} */ (loaded.exports).auditDriverTrace;
}

test.describe("Native driver worker-directory proof", () => {
  test("a controller inherits only the explicitly recorded launch cwd", () => {
    assert.equal(directoryTracker("/work")(start), "/work");
  });
  test("a forked worker must change directory before exec", () => {
    const track = directoryTracker("/work"); track(start); track(fork);
    assert.throws(() => track(worker), /neutral cwd/);
  });
  test("the root cwd is inherited by worker-created threads", () => {
    const track = directoryTracker("/work"); track(start); track(fork); track('101 chdir("/") = 0'); track(worker);
    track("101 clone3({flags=CLONE_VM|CLONE_FS|CLONE_THREAD}, 88) = 102");
    assert.equal(track('102 openat(AT_FDCWD, "/usr/lib/libz.so.1", O_RDONLY) = 3'), "/");
    assert.equal(track('100 openat(AT_FDCWD, "/etc/hosts", O_RDONLY) = 3'), "/work");
  });
  test("CLONE_FS changes are shared, whereas a fork keeps a separate cwd", () => {
    const track = directoryTracker("/work"); track(start); track(fork);
    track("100 clone3({flags=CLONE_FS}, 88) = 102"); track('102 chdir("/") = 0');
    assert.equal(track('100 openat(AT_FDCWD, "/etc/hosts", O_RDONLY) = 3'), "/");
    assert.equal(track('101 openat(AT_FDCWD, "/etc/hosts", O_RDONLY) = 3'), "/work");
  });
  test("an untraced thread cannot assert a worker directory", () => {
    const track = directoryTracker("/work"); track(start);
    assert.throws(() => track('200 chdir("/") = 0'), /ancestry/);
  });
  test("a worker marker in argv is not environment evidence", () => {
    const track = directoryTracker("/work"); track(start); track(fork); track('101 chdir("/") = 0');
    assert.throws(() => track('101 execve("/mom", ["/mom", "MOM_CLUSTER_WORKER=1"], []) = 0'), /neutral cwd/);
  });
  test("an absent, false or partial worker marker is rejected", () => {
    for (const marker of ["", "0", "1extra"]) {
      const track = directoryTracker("/work"); track(start); track(fork); track('101 chdir("/") = 0');
      assert.throws(() => track(worker.replace("WORKER=1", "WORKER=" + marker)), /neutral cwd/);
    }
  });
  test("failed exec does not prove an initialized worker", () => {
    const track = directoryTracker("/work"); track(start); track(fork); track('101 chdir("/") = 0');
    assert.throws(() => track(worker.replace("= 0", "= -1 ENOENT")), /neutral cwd/);
  });
  test("a failed or non-root chdir is not accepted", () => {
    for (const call of ['100 chdir("/") = -1 EACCES', '100 chdir("/tmp") = 0']) {
      const track = directoryTracker("/work"); track(start); assert.throws(() => track(call), /cwd change/);
    }
  });
  test("descriptor-based cwd changes fail closed", () => {
    const track = directoryTracker("/work"); track(start); assert.throws(() => track("100 fchdir(3) = 0"), /Descriptor/);
  });
  test("unfinished clone evidence cannot imply child inheritance", () => {
    const track = directoryTracker("/work"); track(start);
    assert.throws(() => track("100 clone3({flags=CLONE_FS}, 88 <unfinished ...>"), /Incomplete/);
  });
  test("PID reuse cannot overwrite an earlier worker's ancestry", () => {
    const track = directoryTracker("/work"); track(start); track(fork); assert.throws(() => track(fork), /Reused/);
  });
  test("missing PIDs and ambiguous launch directories are rejected", () => {
    assert.throws(() => directoryTracker("/work")(start.replace("100 ", "")), /process identity/);
    for (const cwd of ["work", "/work/..", "/work//cache"]) {assert.throws(() => directoryTracker(cwd), /absolute launch/);}
  });
});

test.describe("Native driver searches and unchanged application boundaries", () => {
  const closure = {
    defaultSearch: ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"],
    roots: [{requested: "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1"}], lateRoots: [],
    libraries: [{file: "/usr/lib/x86_64-linux-gnu/libdriver.so.1.2", soname: "libdriver.so.1",
      needed: ["libz.so.1"], search: ["$ORIGIN/../lib:"]}],
  };
  test("driver entrypoint and SONAME misses include ordinary OS loader searches", () => {
    const searches = driverSearches(closure);
    assert.ok(searches.has("/lib/x86_64-linux-gnu/glibc-hwcaps/x86-64-v4/libze_intel_gpu.so.1"));
    assert.ok(searches.has("/usr/lib/x86_64-linux-gnu/libdriver.so.1"));
    assert.ok(searches.has("/lib/x86_64-linux-gnu/glibc-hwcaps/x86-64-v3/libz.so.1"));
  });
  test("origin-dependent misses account for the loader's lib/usr-lib path spelling", () => {
    const searches = driverSearches(closure);
    assert.ok(searches.has("/lib/x86_64-linux-gnu/../lib/libz.so.1"));
    assert.ok(searches.has("/usr/lib/x86_64-linux-gnu/../lib/libz.so.1"));
    assert.ok(searches.has("libz.so.1"));
  });
  test("metadata-derived misses cannot introduce arbitrary application libraries", () => {
    const searches = driverSearches(closure);
    assert.equal(searches.has("/usr/lib/x86_64-linux-gnu/libsycl.so.8"), false);
    assert.equal(searches.has("/tmp/libz.so.1"), false);
    assert.throws(() => auditNativeTrace(start + '\n100 openat(AT_FDCWD, "/usr/lib/libdriver.so.1", O_RDONLY) = 3\n', "/mom"), /external runtime/);
  });
  test("a successfully opened OS alias proves the canonical inspected driver was loaded", () => {
    const result = aliasAudit()(start + '\n100 openat(AT_FDCWD, "/lib/libze_intel_gpu.so.1.2", O_RDONLY) = 3\n',
      "/mom", ["/usr/lib/libze_intel_gpu.so.1"], {launchDirectory: "/", processes: 1});
    assert.ok(result.opened.includes("/lib/libze_intel_gpu.so.1.2"), "retain the literal observed path");
  });
  test("opening only a driver dependency does not prove its entrypoint was loaded", () => {
    assert.throws(() => aliasAudit()(start + '\n100 openat(AT_FDCWD, "/lib/libsupport.so.1", O_RDONLY) = 3\n',
      "/mom", ["/usr/lib/libze_intel_gpu.so.1"], {launchDirectory: "/", processes: 1}), /No inspected GPU driver/);
  });
  const vaTrace = start + '\n100 openat(AT_FDCWD, "/lib/libigdrcl.so", O_RDONLY) = 3\n' +
    '100 openat(AT_FDCWD, "/lib/libva.so.2", O_RDONLY) = -1 ENOENT (No such file or directory)\n';
  test("Intel OpenCL permits an absent optional media-sharing probe, not a new dependency", () => {
    assert.doesNotThrow(() => aliasAudit(true)(vaTrace, "/mom", ["/usr/lib/libigdrcl.so"], {launchDirectory: "/", processes: 1}));
  });
  test("an optional media probe must actually be absent in the audited filesystem view", () => {
    assert.throws(() => aliasAudit(true, true)(vaTrace, "/mom", ["/usr/lib/libigdrcl.so"], {launchDirectory: "/", processes: 1}), /Unproven driver library search/);
  });
  test("successful media-library loads still need independent closure inspection", () => {
    assert.throws(() => aliasAudit(true)(vaTrace.replace("-1 ENOENT (No such file or directory)", "4"), "/mom",
      ["/usr/lib/libigdrcl.so"], {launchDirectory: "/", processes: 1}), /outside the inspected system-driver closure/);
  });
  test("media probes are not an exemption for unrelated GPU backends", () => {
    assert.throws(() => aliasAudit()(vaTrace.replaceAll("libigdrcl.so", "libze_intel_gpu.so.1.2"), "/mom",
      ["/usr/lib/libze_intel_gpu.so.1"], {launchDirectory: "/", processes: 1}), /Unproven driver library search/);
  });
  test("origin searches retain symlink-sensitive dot components", () => {
    assert.equal(librarySearches(["$ORIGIN/../lib"], "/usr/lib/vendor", "libz.so.1")[0], "/usr/lib/vendor/../lib/libz.so.1");
  });
  test("an empty ELF search component remains relative, not origin-relative", () => {
    assert.deepEqual(librarySearches([""], "/usr/lib/vendor", "libz.so.1"), ["libz.so.1",
      "glibc-hwcaps/x86-64-v4/libz.so.1", "glibc-hwcaps/x86-64-v3/libz.so.1", "glibc-hwcaps/x86-64-v2/libz.so.1"]);
  });
  test("the ELF braced origin form preserves its original spelling after expansion", () => {
    // eslint-disable-next-line no-template-curly-in-string -- Literal ELF syntax.
    assert.equal(librarySearches(["${ORIGIN}/driver"], "/usr/lib/vendor", "libz.so.1")[0], "/usr/lib/vendor/driver/libz.so.1");
  });
  test("no search entry implicitly introduces a current-directory exception", () => {
    assert.deepEqual(librarySearches([], "/usr/lib/vendor", "libz.so.1"), []);
  });
  test("only exact OS loader configuration paths are data, not SO imports", () => {
    auditNativeTrace(start + '\n100 openat(AT_FDCWD, "/etc/ld.so.conf", O_RDONLY) = 3\n' +
      '100 openat(AT_FDCWD, "/etc/ld.so.conf.d", O_RDONLY|O_DIRECTORY) = 4\n' +
      '100 openat(AT_FDCWD, "/etc/ld.so.conf.d/cuda-13.conf", O_RDONLY) = 5\n', "/mom");
  });
  test("a similarly named application, traversal or unexpected OS child is still rejected", () => {
    for (const file of ["/tmp/ld.so.conf", "/etc/ld.so.conf.d/../evil.so", "/etc/ld.so.conf.d/evil.so", "/etc/ld.so.conf.d/evil.conf.so"]) {
      assert.throws(() => auditNativeTrace(start + `\n100 openat(AT_FDCWD, "${file}", O_RDONLY) = 3\n`, "/mom"), /external runtime/);
    }
  });
  test("closure reporting does not admit compiler dependencies to application ELF imports", () => {
    assert.throws(() => auditNativeTrace(start + '\n100 openat(AT_FDCWD, "/usr/lib/libigc.so.2", O_RDONLY) = 3\n', "/mom"), /external runtime/);
  });
  test("missing trace or an invalid driver root fails before inspecting the host", () => {
    assert.throws(() => auditDriverTrace("", "/mom", [], {launchDirectory: "/work"}), /Missing/);
    assert.throws(() => auditDriverTrace(start, "/mom", ["/tmp/evil.so"], {launchDirectory: "/work"}), /unique absolute system GPU/);
  });
});
