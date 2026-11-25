"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");
const {tempDir} = require("./common/temp-dir");
const {extractCppBlock} = require("./common/cpp-source");

const sourceFile = path.join(__dirname, "..", "xmrig", "hw", "msr", "Msr_linux.cpp");
const source = fs.readFileSync(sourceFile, "utf8");

// Members of the vendored MsrPrivate class, anchored at its four-space member indentation.
/** @param {string} declaration */
const sourceMember = declaration => extractCppBlock(source, "    " + declaration);

test.describe("Native Linux MSR helper boundary", {skip: process.platform !== "linux"}, () => {
  test("uses prepared MSR access without shelling out; Engineering keeps its mocked fallback", async t => {
    const root = tempDir(t, "mom-native-msr-");

    const allowWrites = sourceMember("inline bool msr_allow_writes()").replace(
      '"/sys/module/msr/parameters/allow_writes"', '"/proc/self/fd/3"');
    assert.notEqual(allowWrites, sourceMember("inline bool msr_allow_writes()"), "fixture must redirect only the sysfs path");
    const fixture = path.join(root, "msr-policy.cpp");
    const body = [
      "#include <cstdlib>", "#include <cstring>", "#include <fcntl.h>", "#include <fstream>",
      "#include <string>", "#include <sys/mman.h>", "#include <unistd.h>",
      "static int fakeSystemCalls = 0;",
      "static bool fakeDeviceAvailable = false;",
      "[[maybe_unused]] static int fakeSystem(const char *command) {",
      "  ++fakeSystemCalls;",
      '  return std::strcmp(command, "/sbin/modprobe msr allow_writes=on > /dev/null 2>&1") == 0 ? 0 : 9;',
      "}",
      "static int msr_open(int32_t cpu, int flags) {",
      "  if (!fakeDeviceAvailable || cpu != -1 || flags != O_RDWR) return -1;",
      "  return open(\"/dev/null\", O_RDWR);",
      "}",
      "static bool installParameter(const char *value) {",
      "  const int fd = memfd_create(\"msr-allow-writes\", MFD_CLOEXEC | MFD_ALLOW_SEALING);",
      "  if (fd < 0) return false;",
      "  const size_t length = std::strlen(value);",
      "  if (write(fd, value, length) != static_cast<ssize_t>(length)) return false;",
      "  if (fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) < 0) return false;",
      "  if (fd != 3) { if (dup2(fd, 3) < 0) return false; close(fd); }",
      "  return true;",
      "}",
      "#define system fakeSystem",
      "class MsrPrivate {",
      "public:",
      sourceMember("inline MsrPrivate()"),
      "  bool isAvailable() const { return m_available; }",
      allowWrites,
      sourceMember("inline bool msr_device_available()"),
      sourceMember("inline bool msr_modprobe()"),
      "private:",
      "  const bool m_available;",
      "};",
      "int main(int argc, char **argv) {",
      "  if (argc != 5) return 2;",
      "  close(3);",
      '  if (std::strcmp(argv[1], "--unreadable") != 0 && !installParameter(argv[1])) return 4;',
      '  fakeDeviceAvailable = std::strcmp(argv[4], "1") == 0;',
      "  MsrPrivate msr;",
      '  const bool expected = std::strcmp(argv[2], "1") == 0;',
      "  const int expectedSystemCalls = std::atoi(argv[3]);",
      "  return msr.isAvailable() == expected && fakeSystemCalls == expectedSystemCalls ? 0 : 3;",
      "}",
    ].join("\n") + "\n";
    fs.writeFileSync(fixture, body);

    const compiler = process.env["CXX"] || "c++";
    /** @param {string} output @param {boolean} nativeRuntime */
    const compile = (output, nativeRuntime) => {
      const result = spawnSync(compiler, ["-std=c++20", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror", ...(nativeRuntime ? ["-DMOM_NATIVE_RUNTIME"] : []), fixture, "-o", output],
        {encoding: "utf8", timeout: 15000, maxBuffer: 1 << 20});
      assert.equal(result.status, 0, [result.error?.message, result.stdout, result.stderr].filter(Boolean).join("\n"));
      assert.equal(result.signal, null);
    };
    const native = path.join(root, "native-msr-policy"), engineering = path.join(root, "engineering-msr-policy");
    compile(native, true); compile(engineering, false);

    /** @param {string} binary @param {string} state @param {boolean} device @param {boolean} available @param {number} systemCalls */
    const run = (binary, state, device, available, systemCalls) => {
      const result = spawnSync(binary, [state, available ? "1" : "0", String(systemCalls), device ? "1" : "0"],
        {encoding: "utf8", timeout: 5000, maxBuffer: 1 << 20});
      assert.equal(result.status, 0, [result.error?.message, result.stdout, result.stderr].filter(Boolean).join("\n"));
      assert.equal(result.signal, null);
    };

    /** @type {Array<[string, string, string, boolean, boolean, number]>} */
    const scenarios = [
      ["accepts canonical on", native, "on\n", false, true, 0],
      ["accepts canonical default", native, "default\n", false, true, 0],
      ["rejects off even when the device opens", native, "off\n", true, false, 0],
      ["rejects unknown state even when the device opens", native, "unexpected\n", true, false, 0],
      ["rejects an empty state even when the device opens", native, "\n", true, false, 0],
      ["rejects a trailing token even when the device opens", native, "on garbage\n", true, false, 0],
      ["probes the device only if the parameter cannot be opened", native, "--unreadable", true, true, 0],
      ["fails closed if the unreadable parameter has no device", native, "--unreadable", false, false, 0],
      ["retains the mocked Engineering fallback", engineering, "--unreadable", false, true, 1],
    ];
    for (const [name, binary, state, device, available, systemCalls] of scenarios) {
      await t.test(name, () => run(binary, state, device, available, systemCalls));
    }
  });
});
