"use strict";

// Build partitions, NOT a compiler-library registry: upstream Ninja still discovers every
// transitive dependency and the final proof owns the complete static link closure. Split the
// cold compiler build across short CI jobs, carrying only explicit same-run build artifacts.
const fs = require("node:fs"), path = require("node:path");
const {isDeepStrictEqual} = require("node:util");
const {fileHash} = require("./build-source-inputs");
const stageTargets = Object.freeze({
  support: Object.freeze(["LLVMSupport", "LLVMCore", "LLVMAnalysis", "LLVMObject", "LLVMMC"]),
  targets: Object.freeze(["LLVMX86CodeGen", "LLVMNVPTXCodeGen", "LLVMSPIRVCodeGen", "LLVMPasses", "LLVMSPIRVLib"]),
  frontend: Object.freeze(["clangSema", "clangAST", "clangSerialization", "clangParse"]),
  link: Object.freeze(["native-dpcpp-jit"]),
});
const jitBuildStages = Object.freeze(Object.keys(stageTargets));

/** Substitute only the target partition in the shared original compiler build plan. Stage
 * completion is not SDK certification; the final producer must still inspect/recheck inputs.
 * @param {ReturnType<typeof import('./native-jit-sdk').jitSdkPlan>} plan @param {string} stage */
function jitBuildStage(plan, stage) {
  const expected = [plan?.compile?.[0], "--build", plan?.build, "--target", "native-dpcpp-jit", "--parallel", plan?.compile?.[6], "--", "-k", "0"];
  if (!plan || !["linux", "windows"].includes(plan.target) || !path.isAbsolute(plan.build) ||
      !Object.hasOwn(stageTargets, stage) || !Array.isArray(plan.compile) || plan.compile.length !== expected.length ||
      !plan.compile.every((value, index) => value === expected[index]) || !path.isAbsolute(plan.compile[0] || "") || !/^(?:[1-9]|1[0-6])$/.test(plan.compile[6] || "")) {
    throw new Error("Select a supported original JIT plan and named CI build stage");
  }
  return [...plan.compile.slice(0, 4), ...Reflect.get(stageTargets, stage), ...plan.compile.slice(5)];
}

/** Bind generated objects/tools as well as sources between same-workflow CI jobs. Ninja's
 * command log alone cannot detect a changed object in a transferred build tree. Walk only
 * this explicit output, never follow links, and bound metadata independently of file sizes.
 * @param {string} build @returns {Record<string,string>} */
function jitBuildOutputs(build) {
  if (!path.isAbsolute(build) || !fs.lstatSync(build).isDirectory()) {throw new Error("Select a real absolute compiler build directory");}
  /** @type {Record<string,string>} */ const files = {};
  let entries = 0, metadata = 0;
  /** @param {string} directory @param {number} depth */
  function visit(directory, depth) {
    if (depth > 32) {throw new Error("Compiler build directory nesting exceeds bounds");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), name = path.relative(build, file).split(path.sep).join("/");
      if (++entries > 65536 || name.length > 4096 || Array.from(name).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127)) {throw new Error("Compiler build inventory exceeds bounds");}
      if (entry.isDirectory()) {visit(file, depth + 1); continue;}
      let value;
      if (entry.isFile()) {value = "mode:" + (fs.statSync(file).mode & 0o777).toString(8) + ":sha256:" + fileHash(file);} else if (entry.isSymbolicLink()) {
        const link = fs.readlinkSync(file);
        if (link.length > 4096 || Array.from(link).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127)) {throw new Error("Invalid compiler output link");}
        value = "link:" + link;
      } else {throw new Error("Compiler build output is not a regular file or link");}
      if ((metadata += Buffer.byteLength(name) + Buffer.byteLength(value) + 16) > (6 << 20)) {throw new Error("Compiler build metadata exceeds bounds");}
      files[name] = value;
    }
  }
  visit(build, 0);
  if (!Object.keys(files).length) {throw new Error("Compiler build output inventory is empty");}
  return files;
}

/** Refuse changed intermediate objects or source/tool receipts before executing a resumed
 * compiler. This is build continuity, not runtime or publication qualification.
 * @param {unknown} value
 * @param {{stage: string, beforeSha256: string, configuredSha256: string, commandsSha256: string,
 * inputs: Record<string,string>, outputs: Record<string,string>}} expected @param {string} build */
function jitBuildHandoff(value, expected, build) {
  const prior = /** @type {{version?: number, buildOnly?: boolean, libraries?: string[]} & typeof expected | null} */ (value);
  if (!prior || prior.version !== 1 || prior.buildOnly !== true || !["configure", ...jitBuildStages].includes(expected.stage) ||
      !Object.entries(expected).every(([key, data]) => isDeepStrictEqual(Reflect.get(prior, key), data)) ||
      !Array.isArray(prior.libraries) || !prior.libraries.length || prior.libraries.length > 512 || new Set(prior.libraries).size !== prior.libraries.length ||
      prior.libraries.some(file => typeof file !== "string" || !path.isAbsolute(file) || !file.startsWith(build + path.sep) ||
        path.normalize(file) !== file || !/\.(?:a|lib)$/.test(file))) {
    throw new Error("Compiler SDK phase handoff is missing, stale or changed; do not restamp or use a persistent cache");
  }
  return prior.libraries;
}

module.exports = {jitBuildStages, jitBuildStage, jitBuildOutputs, jitBuildHandoff};
