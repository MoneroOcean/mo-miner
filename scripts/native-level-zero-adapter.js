"use strict";

// Build-time namespace isolation for the original V2 adapter, sharing the existing compiler
// isolation primitive. No driver shims, copied API registry, GPU discovery or execution.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isolateCompilerObject} = require("./native-compiler-isolation");
const {fileHash} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");

/** @param {string} archive @param {string} compiler @param {string} headers @param {string} output */
function isolateLevelZeroAdapter(archive, compiler, headers, output) {
  if (process.platform !== "linux" || process.env["MOM_AOT_BUILD_ONLY"] !== "1" ||
      ![archive, compiler, headers, output].every(path.isAbsolute) || !isStaticArchive(archive)) {
    throw new Error("Static Level Zero isolation requires Linux, MOM_AOT_BUILD_ONLY=1 and explicit archive/compiler/header/output paths");
  }
  fs.mkdirSync(output, {recursive: true});
  const attempt = fs.mkdtempSync(path.join(output, "attempt-")), repo = path.resolve(__dirname, ".."),
    factory = path.join(repo, "tests/native/dpcpp-static/level-zero-v2-factory.cpp");
  const files = [archive, compiler, factory, path.join(headers, "unified-runtime/ur_api.h"), path.join(headers, "unified-runtime/ur_ddi.h")];
  const inputs = Object.fromEntries(files.map(file => [file, fileHash(file)]));
  /** @param {string} command @param {string[]} args @param {string} label */
  function run(command, args, label) {
    fs.writeFileSync(path.join(attempt, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
    const fd = fs.openSync(path.join(attempt, label + ".log"), "wx");
    try {
      const result = spawnSync(command, args, {stdio: ["ignore", fd, fd]});
      if (result.error || result.signal || result.status !== 0) {throw new Error("Static Level Zero stage failed: " + label, {cause: result.error});}
    } finally {fs.closeSync(fd);}
  }
  const bridge = path.join(attempt, "factory.o"), raw = path.join(attempt, "adapter-raw.o");
  run(compiler, ["-std=c++17", "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", "-isystem", headers, "-c", factory, "-o", bridge], "factory");
  run("ld", ["-r", "--whole-archive", archive, "--no-whole-archive", bridge, "-o", raw], "partial-link");
  const isolated = isolateCompilerObject(raw, path.join(attempt, "isolated"),
    {prefix: "mom_l0_v2_", entryPoints: ["mom_ur_level_zero_v2_factory"]});
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("Level Zero isolation input changed: " + file);}}
  fs.writeFileSync(path.join(attempt, "result.json"), JSON.stringify({...isolated, inputs,
    scope: "Original Level Zero V2 adapter, private definitions/COMDAT keys, one opaque dispatch factory. Build only; no GPU/runtime qualification."}, null, 2) + "\n", {flag: "wx"});
  const pending = path.join(output, "adapter.o.pending-" + process.pid);
  fs.copyFileSync(isolated.object, pending, fs.constants.COPYFILE_EXCL); fs.renameSync(pending, path.join(output, "adapter.o"));
  return isolated;
}
if (require.main === module) {
  try {
    const [archive, compiler, headers, output, extra] = process.argv.slice(2);
    if (!archive || !compiler || !headers || !output || extra) {throw new Error("Usage: native-level-zero-adapter.js /archive.a /clang++ /UR/include /new-output");}
    isolateLevelZeroAdapter(archive, compiler, headers, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Level Zero isolation failed"); process.exitCode = 1;}
}
module.exports = {isolateLevelZeroAdapter};
