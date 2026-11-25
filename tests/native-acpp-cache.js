"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {describe, test} = require("node:test");
const assert = require("node:assert/strict");
const {tempDir} = require("./common/temp-dir");

const repo = path.resolve(__dirname, "..");
const patchFile = path.join(repo, "runtime/adaptivecpp-static.patch");

/** @param {string} patch @param {string} file */
function patchSection(patch, file) {
  const start = patch.indexOf(`diff --git a/${file} b/${file}`);
  assert.notEqual(start, -1, `missing ${file} patch section`);
  const end = patch.indexOf("\ndiff --git ", start + 1);
  return patch.slice(start, end < 0 ? patch.length : end);
}

/** @param {string} section @param {string} marker */
function patchHunk(section, marker) {
  const lines = section.split(/\r?\n/);
  const start = lines.findIndex(line => line.startsWith("@@ ") && line.includes(marker));
  assert.notEqual(start, -1, `missing patch hunk for ${marker}`);
  let end = start + 1;
  while (end < lines.length) {
    const line = lines[end];
    if (!line || line.startsWith("@@ ") || line.startsWith("diff --git ")) {break;}
    end++;
  }
  return lines.slice(start + 1, end);
}

/** @param {string[]} lines @param {"+" | "-"} marker */
function changedSide(lines, marker) {
  const body = lines.filter(line => line && (line[0] === marker || line[0] === " "))
    .map(line => line.slice(1));
  return body.join("\n");
}

/** @returns {string | null} */
function compiler() {
  const command = process.env["CXX"] || "c++";
  const probe = spawnSync(command, ["--version"], {encoding: "utf8", timeout: process.platform === "win32" ? 30000 : 5000, maxBuffer: 1 << 20});
  const error = /** @type {NodeJS.ErrnoException | undefined} */ (probe.error);
  if (error?.code === "ENOENT") {return null;}
  assert.equal(probe.status, 0, probe.stderr || probe.error?.message || "C++ compiler probe failed");
  return command;
}

describe("AdaptiveCpp atomic cache writes", () => {
  test("a failed optional write returns without throwing and cleans its temporary file", {timeout: process.platform === "win32" ? 90000 : 30000}, t => {
    const command = compiler();
    if (!command) {t.skip("a host C++17 compiler is required"); return;}

    const patch = fs.readFileSync(patchFile, "utf8");
    const filesystem = patchSection(patch, "src/common/filesystem.cpp");
    const functionHunk = patchHunk(filesystem, "bool atomic_write(");
    const beforeTail = changedSide(functionHunk, "-");
    const afterTail = changedSide(functionHunk, "+");
    assert.match(beforeTail, /fs::rename\(tmp_path, p\);/);
    assert.match(afterTail, /fs::rename\(tmp_path, p, rename_error\)/);
    assert.match(afterTail, /fs::remove\(tmp_path, cleanup_error\)/);

    // The unified diff carries only the changed tail; keep the same small,
    // unchanged file/opening scaffold around both versions of that tail.
    const functionScaffold = `
bool atomic_write(const std::string &filename, const std::string &data) {
  fs::path p{filename};
  std::string temp_file = std::to_string(random_number<std::size_t>())+".tmp";
  fs::path tmp_path = p.parent_path() / temp_file;
  std::ofstream ostr{tmp_path, std::ios::binary|std::ios::out|std::ios::trunc};
  if (!ostr.is_open()) return false;
`;
    const before = functionScaffold + beforeTail;
    const after = functionScaffold + afterTail;

    const kernelCache = patchSection(patch, "src/runtime/kernel_cache.cpp");
    const patchedKernelCache = changedSide(kernelCache.split(/\r?\n/), "+");
    const failureDiagnostic = patchedKernelCache.indexOf("Could not store JIT result in persistent kernel cache");
    const failedWriteReturn = patchedKernelCache.indexOf("return;", failureDiagnostic);
    const appdbStore = patchedKernelCache.indexOf("common::filesystem::persistent_storage::get()", failedWriteReturn);
    assert.ok(failureDiagnostic >= 0 && failedWriteReturn > failureDiagnostic && appdbStore > failedWriteReturn,
      "a failed optional cache write must return before the following appdb store operation");

    /** @param {string} source @param {string} name */
    const renameFunction = (source, name) => source.replace("bool atomic_write(", `bool ${name}(`);
    const harness = `
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <system_error>
namespace fs = std::filesystem;
template<class T> T random_number() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen{rd()};
  thread_local std::uniform_int_distribution<T> distribution{0};
  return distribution(gen);
}
${renameFunction(before, "atomic_write_before")}
${renameFunction(after, "atomic_write_after")}
bool equals(const fs::path &file, const std::string &expected) {
  std::ifstream input(file, std::ios::binary);
  return input && std::string(std::istreambuf_iterator<char>(input), {}) == expected;
}
std::size_t entries(const fs::path &directory) {
  return static_cast<std::size_t>(std::distance(fs::directory_iterator(directory), fs::directory_iterator{}));
}
int main(int argc, char **argv) {
  const fs::path root{argv[1]};
  const fs::path ordinary = root / "ordinary";
  fs::create_directories(ordinary);
  const fs::path fresh = ordinary / "fresh.jit";
  if (!atomic_write_after(fresh.string(), "first") || !equals(fresh, "first")) return 10;
  const fs::path existing = ordinary / "existing.jit";
  { std::ofstream seed(existing); seed << "old"; }
  if (!atomic_write_after(existing.string(), "replacement") || !equals(existing, "replacement")) return 11;
  if (atomic_write_after((root / "missing" / "cache.jit").string(), "data")) return 12;

  const std::string overlong(256, 'x');
  const fs::path oldDir = root / "before";
  const fs::path newDir = root / "after";
  fs::create_directories(oldDir);
  fs::create_directories(newDir);
  const fs::path oldTarget = oldDir / overlong;
  const fs::path newTarget = newDir / overlong;
  bool oldThrew = false, oldReturned = false;
  try { atomic_write_before(oldTarget.string(), "compiled-kernel"); oldReturned = true; }
  catch (const fs::filesystem_error &) { oldThrew = true; }
  bool newThrew = false, newReturned = false, newResult = false;
  try { newResult = atomic_write_after(newTarget.string(), "compiled-kernel"); newReturned = true; }
  catch (const fs::filesystem_error &) { newThrew = true; }
  const auto oldLeftovers = entries(oldDir), newLeftovers = entries(newDir);
  std::cout << "before: threw=" << oldThrew << " returned=" << oldReturned << " leftovers=" << oldLeftovers << "\\n"
            << "after: threw=" << newThrew << " returned=" << newReturned << " result=" << newResult << " leftovers=" << newLeftovers << "\\n"
            << "ordinary: fresh=1 replace=1 missing=0\\n";
  if (!oldThrew || oldReturned || oldLeftovers != 1 || newThrew || !newReturned || newResult || newLeftovers != 0) return 13;
  return 0;
}
`;

    const root = tempDir(t, "mom-acpp-cache-");
    const source = path.join(root, "cache-write.cpp"), executable = path.join(root, "cache-write");
    fs.writeFileSync(source, harness, {flag: "wx"});
    const built = spawnSync(command, ["-std=c++17", "-O0", "-g0", source, "-o", executable],
      {encoding: "utf8", timeout: process.platform === "win32" ? 60000 : 20000, maxBuffer: 1 << 20});
    assert.equal(built.status, 0, built.error?.message || built.stderr || "C++ harness compilation failed");
    const result = spawnSync(executable, [root], {encoding: "utf8", timeout: 5000, maxBuffer: 1 << 20});
    assert.equal(result.status, 0, result.stderr || result.stdout || result.error?.message || "C++ harness failed");
    assert.match(result.stdout, /before: threw=1 returned=0 leftovers=1/);
    assert.match(result.stdout, /after: threw=0 returned=1 result=0 leftovers=0/);
    assert.match(result.stdout, /ordinary: fresh=1 replace=1 missing=0/);
  });
});
