"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {test} = require("node:test");

const seedPath = process.env["MOM_C29_SEED_SOURCE"] || path.join(__dirname, "../sycl/c29/seed.inc");
const seed = fs.readFileSync(seedPath, "utf8");
const loop = "for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {";
const guard = "#if defined(_WIN32) && defined(MOM_SYCL_ADAPTIVECPP) && defined(MOM_SYCL_HAS_HIP)";
const pragma = "#pragma clang loop unroll_count(2)";

function seedUnrollDirectives() {
  const lines = seed.split(/\r?\n/);
  const directSeed = lines.findIndex(line => line.trim() === "if (direct_seed) {");
  assert.ok(directSeed >= 0, "Expected the direct C29 seed path");
  const directSeedEnd = lines.findIndex((line, index) => index > directSeed && line.trim() === "} else {");
  assert.ok(directSeedEnd > directSeed, "Expected the fallback seed path after direct seeding");
  const loopLines = lines.flatMap((line, index) =>
    index > directSeed && index < directSeedEnd && line.trim() === loop ? [index] : []);
  assert.equal(loopLines.length, 1, "Expected exactly one C29 seed hash_index loop");

  const loopIndex = loopLines[0];
  assert.ok(loopIndex !== undefined);
  let end = loopIndex - 1;
  while (end >= 0 && lines[end]?.trim() === "") {
    --end;
  }
  assert.equal(lines[end]?.trim(), "#endif", "The pragma guard must end immediately before the seed loop");

  let pragmaIndex = end - 1;
  while (pragmaIndex >= 0 && lines[pragmaIndex]?.trim().startsWith("//")) {
    --pragmaIndex;
  }
  assert.equal(lines[pragmaIndex]?.trim(), pragma, "The seed loop must use unroll_count(2)");

  let guardIndex = pragmaIndex - 1;
  while (guardIndex >= 0 && lines[guardIndex]?.trim().startsWith("//")) {
    --guardIndex;
  }
  assert.equal(lines[guardIndex]?.trim(), guard, "The seed unroll pragma must be Windows AdaptiveCPP HIP-only");

  const meaningful = lines.slice(guardIndex, end + 1).map(line => line.trim())
    .filter(line => line !== "" && !line.startsWith("//"));
  assert.deepEqual(meaningful, [guard, pragma, "#endif"]);
  const actualGuard = lines[guardIndex], actualPragma = lines[pragmaIndex];
  assert.ok(actualGuard !== undefined && actualPragma !== undefined);
  return {guard: actualGuard.trim(), pragma: actualPragma.trim()};
}

test.describe("C29 seed unroll guard", {concurrency: false}, () => {
  test("keeps the exact HIP-specific unroll pragma adjacent to the direct seed loop", () => {
    seedUnrollDirectives();
  });

  test("preprocesses only for Windows AdaptiveCPP HIP",
    {skip: process.platform === "win32" ? "c++ preprocessing fixture is Linux-only" : false}, () => {
      const {guard: actualGuard, pragma: actualPragma} = seedUnrollDirectives();
      const marker = "int mom_c29_seed_unroll_active;";
      const fixture = `${actualGuard}\n${actualPragma}\n${marker}\n#endif\n`;
      const cases = [
        {name: "Windows AdaptiveCPP HIP", defines: ["_WIN32", "MOM_SYCL_ADAPTIVECPP", "MOM_SYCL_HAS_HIP"], active: true},
        {name: "Linux AdaptiveCPP HIP", defines: ["MOM_SYCL_ADAPTIVECPP", "MOM_SYCL_HAS_HIP"], active: false},
        {name: "Windows HIP without AdaptiveCPP", defines: ["_WIN32", "MOM_SYCL_HAS_HIP"], active: false},
        {name: "Windows CUDA", defines: ["_WIN32", "MOM_SYCL_ADAPTIVECPP", "MOM_SYCL_HAS_CUDA"], active: false},
        {name: "Windows oneAPI", defines: ["_WIN32", "MOM_SYCL_HAS_ONEAPI"], active: false},
      ];

      for (const item of cases) {
        const result = spawnSync("c++", [
          ...item.defines.map(name => `-D${name}=1`), "-E", "-P", "-x", "c++", "-",
        ], {input: fixture, encoding: "utf8", timeout: 10000});
        assert.equal(result.status, 0, `${item.name} preprocessing failed: ${result.stderr}`);
        assert.equal(result.stdout.includes(actualPragma), item.active, `${item.name} pragma result`);
        assert.equal(result.stdout.includes(marker), item.active, `${item.name} guard result`);
      }
    });
});
