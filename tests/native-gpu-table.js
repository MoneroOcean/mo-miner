"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {gpuDeclarations, gpuTable} = require("../scripts/native-gpu-table");
const original = fs.readFileSync(path.join(__dirname, "../sycl/lib.h"), "utf8");

test.describe("Native GPU table follows the maintained interface", () => {
  test("every original declaration produces one typed table field", () => {
    const {entries, header} = gpuTable(original, "mom_native_gpu_dpcpp");
    assert.equal(entries.length, 27);
    for (const {name} of entries) {assert.ok(header.includes(`decltype(&::${name}) ${name};`));}
  });
  test("nested template commas do not split parameter names", () => {
    const [entry] = gpuDeclarations("MOM_SYCL_API std::map<std::string, std::string> query(const std::map<std::string, unsigned>& values, unsigned count);");
    assert.deepEqual(entry?.args, ["values", "count"]);
  });
  test("only the maintained nullptr default is removed from generated definitions", () => {
    const rows = gpuDeclarations(original), pearl = rows.find((row) => row.name === "pearlhash");
    assert.ok(pearl?.args.includes("base_target")); assert.doesNotMatch(pearl?.declaration || "", /nullptr|=/);
    assert.throws(() => gpuDeclarations("MOM_SYCL_API int bad(int value = 42);"), /default argument/);
  });
  test("the fixed jackpot array argument and empty noexcept cleanup retain their ABI", () => {
    const rows = gpuDeclarations(original);
    assert.deepEqual(rows.find((row) => row.name === "pearlhash_claim")?.args, ["jackpot", "adjustment_factor"]);
    assert.equal(rows.find((row) => row.name === "sycl_cleanup")?.declaration, "void sycl_cleanup() noexcept");
  });
  test("added public entries are picked up without another function list", () => {
    assert.ok(gpuTable(original + "\nMOM_SYCL_API unsigned extra(unsigned value);\n", "mom_native_gpu_new").header.includes("decltype(&::extra) extra;"));
  });
  test("missing and duplicated declarations fail before code generation", () => {
    for (const source of ["", "MOM_SYCL_API int missing()", original + "\nMOM_SYCL_API void sycl_cleanup() noexcept;"]) {
      assert.throws(() => gpuDeclarations(source), /Incomplete or duplicate/);
    }
  });
  test("unnamed, function-pointer, expression and duplicate parameters are not guessed", () => {
    for (const source of ["int bad(unsigned)", "int bad(void (*callback)())", "int bad(int* data[SIZE])", "int bad(unsigned value, int value)", "int bad(unsigned value,)"]) {
      assert.throws(() => gpuDeclarations("MOM_SYCL_API " + source + ";"), /unsupported|Unsupported|duplicate|Empty/);
    }
  });
  test("unbalanced template types cannot produce a subtly different table", () => {
    for (const type of ["std::map<int, int>>", "std::map<int, int"]) {
      assert.throws(() => gpuDeclarations(`MOM_SYCL_API int bad(const ${type}& values);`), /Unbalanced|unsupported/);
    }
  });
  test("factory identifiers cannot inject generated source or collide with arbitrary globals", () => {
    for (const name of ["factory", "mom_native_gpu_", "mom_native_gpu_bad;evil()", "mom_native_gpu_Upper", "mom_native_gpu_" + "x".repeat(33)]) {
      assert.throws(() => gpuTable(original, name), /factory name/);
    }
  });
  test("one immutable function table has no runtime loader or device work", () => {
    const {definition} = gpuTable(original, "mom_native_gpu_dpcpp");
    assert.match(definition, /static const mom::runtime::GpuTable table/);
    assert.doesNotMatch(definition, /dlopen|LoadLibrary|new |getenv|sycl::|\.so|\.node/);
  });
});

test.describe("Generated native GPU table C++ ABI", {skip: process.platform === "win32"}, () => {
  let directory = "";
  test.before(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-table-"));
    fs.mkdirSync(path.join(directory, "sycl"));
    const header = "#pragma once\n#include <cstdint>\n#include <string>\n#include <map>\n#define MOM_SYCL_API\n" +
      "MOM_SYCL_API uint64_t work(unsigned a, uint64_t b);\n" +
      "MOM_SYCL_API std::map<std::string, unsigned> query(const std::map<std::string, unsigned>& values);\n" +
      "MOM_SYCL_API const char* claim(uint8_t out[32], uint32_t* factor = nullptr);\n" +
      "MOM_SYCL_API void fail(const std::string& message);\nMOM_SYCL_API void cleanup() noexcept;\n";
    fs.writeFileSync(path.join(directory, "sycl/lib.h"), header);
    const bridge = gpuTable(header, "mom_native_gpu_fixture");
    fs.writeFileSync(path.join(directory, "gpu-table.h"), bridge.header);
    fs.writeFileSync(path.join(directory, "factory.cpp"), bridge.definition);
    fs.writeFileSync(path.join(directory, "fixture.cpp"), '#include "gpu-table.h"\n#include <cassert>\n#include <cstring>\n' +
      "uint64_t work(unsigned a,uint64_t b){return a+b;}\nstd::map<std::string,unsigned> query(const std::map<std::string,unsigned>& v){return v;}\n" +
      'const char* claim(uint8_t out[32],uint32_t* factor){out[31]=37;if(factor)*factor=524288;return "proof";}\n' +
      "void fail(const std::string& message){throw message;}\nvoid cleanup() noexcept{}\n" +
      "int main(){auto table=mom_native_gpu_fixture();assert(table==mom_native_gpu_fixture());\n" +
      "assert(table->work(7,0x100000003ULL)==0x10000000aULL);\n" +
      'assert(table->query({{"value",37}}).at("value")==37);uint8_t bytes[32]{};uint32_t factor=0;\n' +
      'assert(std::strcmp(table->claim(bytes,&factor),"proof")==0&&factor==524288&&bytes[31]==37);\n' +
      'bool caught=false;try{table->fail("cross-boundary");}catch(const std::string& message){caught=message=="cross-boundary";}assert(caught);\n' +
      "static_assert(noexcept(table->cleanup()));table->cleanup();}\n");
    const built = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", directory,
      path.join(directory, "factory.cpp"), path.join(directory, "fixture.cpp"), "-o", path.join(directory, "fixture")], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.error?.message || built.stderr);
  });
  test.after(() => {if (directory) {fs.rmSync(directory, {recursive: true, force: true});}});
  test("scalar width, STL references/return, array/pointer output, exceptions and noexcept survive the typed factory", () => {
    const result = spawnSync(path.join(directory, "fixture"), [], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });
});
