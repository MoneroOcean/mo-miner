"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const llvm = "/usr/lib/llvm-21/bin", patch = fs.readFileSync(path.join(__dirname, "../runtime/adaptivecpp-jit.patch"), "utf8");
/** @param {string} name */
function block(name) {
  const start = patch.indexOf(`diff --git a/${name} b/${name}\n`), end = patch.indexOf("\ndiff --git ", start + 1);
  assert.ok(start >= 0, "Missing original patched source: " + name);
  return patch.slice(start, end < 0 ? undefined : end).split("\n").filter(line => line.startsWith("+") && !line.startsWith("+++")).map(line => line.slice(1)).join("\n");
}

test.describe("Native Windows ELF and PTX symbol identity", {skip: process.platform !== "linux" || !fs.existsSync(path.join(llvm, "llvm-config")), timeout: 30000}, () => {
  let root = "", executable = "";
  const run = (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {string} */ input = "") => spawnSync(command, args,
    {input, encoding: "utf8", timeout: 10000, maxBuffer: 2 << 20});
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-symbols-")); executable = path.join(root, "probe");
    const helper = block("include/hipSYCL/common/string_utils.hpp");
    const translator = block("src/compiler/llvm-to-backend/LLVMToBackend.cpp");
    const begin = translator.indexOf("  std::vector<std::pair<llvm::GlobalValue *, std::string>> names;"), end = translator.indexOf("\n}", begin);
    assert.ok(begin >= 0 && end > begin, "Compile the actual production rename body");
    const source = `#include <string>\n#include <string_view>\n#include <vector>\n#include <cstdio>\n#include <llvm/IR/Module.h>\n#include <llvm/IR/IRBuilder.h>\n#include <llvm/Support/raw_ostream.h>\nnamespace common {\n${helper}\n}\n
int main(int argc, char **argv) {
  if(argc != 2) return 2;
  std::string input{argv[1]};
  if(input.rfind("--", 0) != 0) {
    const auto result = common::device_symbol_name(input);
    std::fwrite(result.data(), 1, result.size(), stdout); return 0;
  }
  llvm::LLVMContext context; llvm::Module FlavoredModule("Windows GPU symbols", context);
  const auto kernelName = std::string{"?kernel@mom@@YAXXZ"};
  auto *type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  auto *kernel = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, kernelName, FlavoredModule);
  kernel->setCallingConv(input.find("ptx") != std::string::npos ? llvm::CallingConv::PTX_Kernel : llvm::CallingConv::AMDGPU_KERNEL);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", kernel)); builder.CreateRetVoid();
  auto *integer = llvm::Type::getInt32Ty(context);
  const auto reserved = common::device_symbol_name(kernelName);
  // Deliberately occupy the kernel's destination first: a one-pass rename auto-suffixes it.
  new llvm::GlobalVariable(FlavoredModule, integer, false, llvm::GlobalValue::ExternalLinkage,
    llvm::ConstantInt::get(integer, 1), reserved);
  new llvm::GlobalVariable(FlavoredModule, integer, false, llvm::GlobalValue::ExternalLinkage,
    llvm::ConstantInt::get(integer, 2), "?data@mom@@3HA");
  if(input.rfind("--after", 0) == 0) {\n${translator.slice(begin, end)}\n}
  FlavoredModule.print(llvm::outs(), nullptr); return 0;
}\n`;
    const file = path.join(root, "probe.cpp"); fs.writeFileSync(file, source);
    const flags = run(path.join(llvm, "llvm-config"), ["--cxxflags", "--ldflags", "--libs", "core", "--system-libs"]);
    assert.equal(flags.status, 0, flags.error?.message || flags.stderr);
    const compile = run(path.join(llvm, "clang++"), [file, ...flags.stdout.trim().split(/\s+/), "-std=c++17", "-O1", "-o", executable]);
    assert.equal(compile.status, 0, compile.error?.message || compile.stderr);
  });
  const expected = (/** @type {string} */ name) => /[^A-Za-z0-9_.$]/.test(name) || name.startsWith("__acpp_elf_") ? "__acpp_elf_" + Buffer.from(name).toString("hex") : name;
  for (const name of ["", "kernel", "_Z17ordinary_kernelv", "__acpp_sscp_kernel", "?kernel@mom@@YAXXZ", "?data@mom@@3HA", "?kernel", "x@y", "x@@y", "@", "__acpp_elf_", "__acpp_elf_3f6b65726e656c406d6f6d4040594158585a", "a.b_42", "kernel@\u00e9", "kernel-\u00e9", "$kernel", "9kernel"]) {
    test("exact shared symbol encoding: " + JSON.stringify(name), () => {const r = run(executable, [name]); assert.equal(r.status, 0, r.stderr); assert.equal(r.stdout, expected(name));}); // eslint-disable-line no-loop-func -- Executable is assigned once in the suite's before hook.
  }
  test("original Windows names fail ELF version parsing; production names link with exact identities", () => {
    for (const fixed of [false, true]) {
      const ir = run(executable, [fixed ? "--after" : "--before"]); assert.equal(ir.status, 0, ir.stderr);
      const object = path.join(root, fixed ? "after.o" : "before.o"), image = path.join(root, fixed ? "after.hsaco" : "before.hsaco");
      const emit = run(path.join(llvm, "llc"), ["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx900", "-filetype=obj", "-o", object, "-"], ir.stdout);
      assert.equal(emit.status, 0, emit.error?.message || emit.stderr);
      const link = run(path.join(llvm, "ld.lld"), ["-shared", "--no-undefined", object, "-o", image]);
      if (!fixed) {assert.notEqual(link.status, 0); assert.match(link.stderr, /has undefined version/); continue;}
      assert.equal(link.status, 0, link.error?.message || link.stderr);
      const kernel = expected("?kernel@mom@@YAXXZ"), reserved = expected(kernel);
      const symbols = run(path.join(llvm, "llvm-nm"), ["--defined-only", image]); assert.equal(symbols.status, 0, symbols.stderr);
      for (const name of [kernel, reserved, expected("?data@mom@@3HA")]) {assert.ok(symbols.stdout.split("\n").some(line => line.endsWith(" " + name)), name);}
      assert.ok(!symbols.stdout.includes(kernel + ".1"), "No order-dependent LLVM rename suffix");
      const metadata = run(path.join(llvm, "llvm-readobj"), ["--notes", image]); assert.equal(metadata.status, 0, metadata.stderr);
      assert.ok(metadata.stdout.includes(kernel + ".kd"), "AMDGPU metadata matches the runtime lookup");
    }
  });
  test("original MSVC names reproduce the PTX abort; production names emit exact kernel and data identities", () => {
    for (const fixed of [false, true]) {
      const ir = run(executable, [fixed ? "--after-ptx" : "--before-ptx"]); assert.equal(ir.status, 0, ir.stderr);
      // The deliberately invalid baseline aborts LLVM: never generate a host core dump.
      const emit = run("/bin/sh", ["-c", 'ulimit -c 0; exec "$@"', "llc", path.join(llvm, "llc"),
        "-mtriple=nvptx64-nvidia-cuda", "-mcpu=sm_80", "-filetype=asm", "-o", "-", "-"], ir.stdout);
      if (!fixed) {assert.equal(emit.error, undefined, "Baseline fails promptly, not by timeout"); assert.notEqual(emit.status, 0); assert.match(emit.stderr, /LLVM ERROR: Symbol name with unsupported characters/); continue;}
      assert.equal(emit.status, 0, emit.error?.message || emit.stderr);
      const kernel = expected("?kernel@mom@@YAXXZ");
      assert.ok(emit.stdout.includes(".entry " + kernel + "("), "PTX entry matches CUDA runtime lookup");
      for (const name of [expected(kernel), expected("?data@mom@@3HA")]) {assert.ok(emit.stdout.includes(" " + name + " ="), name);}
      assert.ok(!emit.stdout.includes(kernel + ".1"), "No order-dependent LLVM rename suffix");
    }
  });
  test("both final translators and SSCP lookups share one mapping without changing Engineering or cache identities", () => {
    for (const backend of ["hip", "cuda"]) {
      const queue = block(`src/runtime/${backend}/${backend}_queue.cpp`);
      assert.match(queue, /#if defined\(_WIN32\) && defined\(ACPP_STATIC_RUNTIME\)\n[\s\S]*common::device_symbol_name\(kernel_name\)/);
      assert.match(queue, /module, image_kernel_name, num_groups/);
    }
    for (const backend of ["amdgpu", "ptx"]) {
      const translator = block(`src/compiler/llvm-to-backend/${backend}/LLVMTo${backend === "amdgpu" ? "Amdgpu" : "Ptx"}.cpp`);
      assert.match(translator, /optimization\.run\(FlavoredModule, modules\);\n#ifdef _WIN32\n {2}normalizeWindowsDeviceSymbols\(FlavoredModule\);\n#endif/);
    }
    assert.equal((patch.match(/^\+.*common::device_symbol_name\(/gm) || []).length, 3);
  });
});
